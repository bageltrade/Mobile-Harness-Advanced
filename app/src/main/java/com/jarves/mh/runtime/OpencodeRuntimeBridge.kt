package com.jarves.mh.runtime

import android.content.Context
import android.util.Log
import com.jarves.mh.model.ChatMessage
import com.jarves.mh.model.ProjectKind
import com.jarves.mh.model.ProviderKind
import com.jarves.mh.model.ProviderProfile
import com.jarves.mh.model.RuntimeEvent
import com.jarves.mh.model.ToolRequest
import java.io.File
import java.io.RandomAccessFile
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.withContext
import org.json.JSONObject

/**
 * Runtime bridge for the official OpenCode CLI (`opencode run`).
 *
 * Headless one-shot mode:
 *   opencode run --format json --auto --model <provider/model> "<prompt>"
 *
 * Installs via npm into the private Linux runtime when selected.
 * Supports OpenCode Zen and BYOK providers through environment credentials.
 */
class OpencodeRuntimeBridge(
    private val context: Context,
    private val secretFor: (ProviderProfile) -> String?,
) : RuntimeBridge {
    private val installer = RuntimeInstaller(context)
    private val checkpoints = WorkspaceCheckpoints(context.filesDir)
    private val eventBus = MutableSharedFlow<RuntimeEvent>(extraBufferCapacity = 64)
    override val events: Flow<RuntimeEvent> = eventBus
    private val finishedSessions = ConcurrentHashMap.newKeySet<String>()
    @Volatile private var activeProcess: Process? = null
    @Volatile private var activeSessionId: String? = null
    @Volatile private var userStopRequested: Boolean = false
    @Volatile private var activeProjectSlug: String? = null

    override suspend fun startSession(
        projectId: String,
        projectSlug: String,
        projectKind: ProjectKind,
        prompt: String,
        conversationHistory: List<ChatMessage>,
        provider: ProviderProfile,
    ): String = withContext(Dispatchers.IO + NonCancellable) {
        val sessionId = UUID.randomUUID().toString()
        finishedSessions.remove(sessionId)
        activeSessionId = sessionId
        userStopRequested = false
        activeProjectSlug = projectSlug
        eventBus.emit(RuntimeEvent.SessionStarted(sessionId))

        val secret = secretFor(provider).orEmpty()
        if (secret.isBlank() && provider.kind != ProviderKind.CLAUDE) {
            eventBus.emit(RuntimeEvent.SessionFailed(sessionId, "No API key is saved for ${provider.kind.title}."))
            return@withContext sessionId
        }

        runCatching {
            check(installer.isAgentInstalled(com.jarves.mh.model.AgentKind.OPENCODE)) {
                "OpenCode is not installed. Open Settings → Coding agent to install it."
            }
            val installed = installer.installedRuntime()
            val workspace = checkpoints.ensureWorkspace(projectId)
            checkpoints.createCheckpoint(projectId, workspace)
            val before = checkpoints.snapshot(workspace)
            val guestWorkspacePath = "/workspace/$projectSlug"

            val modelFlag = resolveModel(provider)
            val environment = linkedMapOf<String, String>().apply {
                putAll(providerEnv(provider, secret))
                // Auto-approve tool use inside the already-sandboxed PRoot guest.
                put("OPENCODE_PERMISSION", "allow")
            }

            val command = buildList {
                add("opencode")
                add("run")
                add("--format")
                add("json")
                add("--auto")
                if (modelFlag.isNotBlank()) {
                    add("--model")
                    add(modelFlag)
                }
                add(prompt)
            }

            Log.d("OpencodeBridge", "Launching: ${command.joinToString(" ")} model=$modelFlag")
            val process = installer.process(
                installed.proot,
                installed.rootfs,
                workspace,
                environment,
                command,
                guestWorkspacePath = guestWorkspacePath,
                emulateHardLinks = false,
            )
            activeProcess = process
            if (userStopRequested) process.destroy()

            val output = StringBuilder()
            val native = process as? NativeSpawnProcess
            var offset = 0L
            while (process.isAlive || (native != null && native.outputFile.length() > offset)) {
                if (native == null) {
                    delay(50)
                    continue
                }
                val available = native.outputFile.length() - offset
                if (available <= 0) {
                    delay(50)
                    continue
                }
                val bytes = ByteArray(minOf(available, 16L * 1024).toInt())
                val count = RandomAccessFile(native.outputFile, "r").use { f ->
                    f.seek(offset)
                    f.read(bytes)
                }
                if (count <= 0) continue
                offset += count
                val chunk = bytes.decodeToString(0, count)
                output.append(chunk)
                parseAndEmit(sessionId, chunk)
            }

            val exit = process.waitFor()
            activeProcess = null
            val changed = checkpoints.changedFiles(workspace, before)
            if (changed.isNotEmpty()) {
                checkpoints.saveChangedPaths(projectId, changed)
                val details = checkpoints.buildChangeDetails(projectId, workspace, checkpoints.readChangedPaths(projectId))
                eventBus.emit(RuntimeEvent.FilesChanged(sessionId, details))
            }

            if (userStopRequested) {
                emitFailureOnce(sessionId, "Stopped by user")
            } else if (exit == 0) {
                emitCompletedOnce(sessionId)
            } else {
                val tail = output.toString().takeLast(800)
                val msg = extractError(tail).ifBlank { "OpenCode exited with code $exit" }
                emitFailureOnce(sessionId, msg)
            }
        }.onFailure { error ->
            Log.e("OpencodeBridge", "Session failed", error)
            emitFailureOnce(sessionId, error.message?.take(500) ?: "OpenCode could not start.")
        }
        activeProcess = null
        activeSessionId = null
        sessionId
    }

    private fun resolveModel(provider: ProviderProfile): String {
        val model = provider.model.ifBlank { provider.kind.defaultModel }
        return when (provider.kind) {
            ProviderKind.OPENCODE_ZEN ->
                if (model.startsWith("opencode/")) model else "opencode/$model"
            ProviderKind.ANTHROPIC ->
                if (model.contains("/")) model else "anthropic/$model"
            ProviderKind.LLM_ROUTER ->
                if (model.contains("/")) model else "openrouter/$model"
            ProviderKind.DEEPSEEK ->
                if (model.contains("/")) model else "deepseek/$model"
            ProviderKind.NVIDIA_NIM ->
                if (model.contains("/")) model else "nvidia/$model"
            ProviderKind.KIMI ->
                if (model.contains("/")) model else "moonshot/$model"
            else -> model
        }
    }

    private fun providerEnv(provider: ProviderProfile, secret: String): Map<String, String> {
        val env = linkedMapOf<String, String>()
        when (provider.kind) {
            ProviderKind.OPENCODE_ZEN -> {
                env["OPENCODE_API_KEY"] = secret
                // Zen also accepts this for the opencode provider id
                env["OPENCODE_ZEN_API_KEY"] = secret
            }
            ProviderKind.ANTHROPIC -> {
                env["ANTHROPIC_API_KEY"] = secret
            }
            ProviderKind.LLM_ROUTER -> {
                env["OPENROUTER_API_KEY"] = secret
            }
            ProviderKind.DEEPSEEK -> {
                env["DEEPSEEK_API_KEY"] = secret
            }
            ProviderKind.NVIDIA_NIM -> {
                env["NVIDIA_API_KEY"] = secret
            }
            ProviderKind.KIMI -> {
                env["MOONSHOT_API_KEY"] = secret
            }
            ProviderKind.CUSTOM -> {
                env["OPENAI_API_KEY"] = secret
                if (provider.resolvedBaseUrl.isNotBlank()) {
                    env["OPENAI_BASE_URL"] = provider.resolvedBaseUrl.trimEnd('/')
                }
            }
            else -> {
                env["OPENCODE_API_KEY"] = secret
            }
        }
        return env
    }

    private suspend fun parseAndEmit(sessionId: String, chunk: String) {
        // OpenCode --format json emits newline-delimited JSON events.
        for (line in chunk.lineSequence()) {
            val trimmed = line.trim()
            if (trimmed.isEmpty()) continue
            if (!trimmed.startsWith("{")) {
                if (trimmed.isNotBlank()) {
                    eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, trimmed + "\n"))
                }
                continue
            }
            runCatching {
                val json = JSONObject(trimmed)
                val type = json.optString("type").ifBlank { json.optString("event") }
                when {
                    type.contains("text", true) || type == "message" || type == "assistant" -> {
                        val text = json.optString("text")
                            .ifBlank { json.optString("content") }
                            .ifBlank { json.optJSONObject("part")?.optString("text").orEmpty() }
                        if (text.isNotBlank()) {
                            eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, text))
                        }
                    }
                    type.contains("tool", true) -> {
                        val name = json.optString("name").ifBlank { json.optString("tool") }.ifBlank { "tool" }
                        eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, "\n[$name]\n"))
                    }
                    type.contains("error", true) -> {
                        val msg = json.optString("message").ifBlank { json.optString("error") }
                        if (msg.isNotBlank()) {
                            eventBus.emit(RuntimeEvent.AssistantDelta(sessionId, "\nError: $msg\n"))
                        }
                    }
                }
            }
        }
    }

    private fun extractError(raw: String): String {
        val lower = raw.lowercase()
        return when {
            "503" in lower || "service unavailable" in lower ->
                "Provider temporarily unavailable (503). Retry the task."
            "401" in lower || "unauthorized" in lower || "invalid api" in lower ->
                "The provider rejected the saved API key."
            "rate limit" in lower || "429" in lower ->
                "Provider rate-limited the request. Try again shortly."
            else -> raw.lineSequence().lastOrNull { it.isNotBlank() }?.take(400).orEmpty()
        }
    }

    private suspend fun emitCompletedOnce(sessionId: String) {
        if (finishedSessions.add(sessionId)) {
            eventBus.emit(RuntimeEvent.SessionCompleted(sessionId))
        }
    }

    private suspend fun emitFailureOnce(sessionId: String, reason: String) {
        if (finishedSessions.add(sessionId)) {
            eventBus.emit(RuntimeEvent.SessionFailed(sessionId, reason))
        }
    }

    override suspend fun respondToApproval(request: ToolRequest, approved: Boolean) {
        // --auto mode does not request interactive approvals.
    }

    override suspend fun stopSession(sessionId: String) = withContext(Dispatchers.IO) {
        if (activeSessionId == sessionId) {
            userStopRequested = true
            activeProcess?.destroy()
            delay(400)
            if (activeProcess?.isAlive == true) activeProcess?.destroyForcibly()
            emitFailureOnce(sessionId, "Stopped by user")
        }
    }

    override suspend fun stopActiveSession() {
        activeSessionId?.let { stopSession(it) }
    }

    fun configureProjectRoot(projectId: String, rootPath: String) {
        checkpoints.configureProjectRoot(projectId, rootPath)
    }

    override suspend fun undoLastChanges(projectId: String): Boolean = withContext(Dispatchers.IO) {
        val checkpoint = checkpoints.checkpointDir(projectId)
        val backup = File(checkpoint, "project")
        if (!backup.isDirectory || !File(checkpoint, "changes.json").isFile) return@withContext false
        val workspace = checkpoints.ensureWorkspace(projectId)
        val paths = checkpoints.readChangedPaths(projectId).filterNot(checkpoints::isInternalRuntimePath)
        if (paths.isEmpty()) return@withContext false
        paths.forEach { path ->
            val target = checkpoints.safeWorkspaceFile(workspace, path)
            val original = checkpoints.safeWorkspaceFile(backup, path)
            if (original.isFile) {
                target.parentFile?.mkdirs()
                original.copyTo(target, overwrite = true)
            } else if (target.isFile) {
                target.delete()
            }
        }
        checkpoint.deleteRecursively()
        true
    }

    override suspend fun acceptLastChanges(projectId: String) {
        withContext(Dispatchers.IO) {
            checkpoints.checkpointDir(projectId).deleteRecursively()
        }
    }

    override suspend fun loadPendingChanges(projectId: String): List<com.jarves.mh.model.ChangeItem> =
        withContext(Dispatchers.IO) {
            val workspace = checkpoints.ensureWorkspace(projectId)
            val paths = checkpoints.readChangedPaths(projectId).filterNot(checkpoints::isInternalRuntimePath)
            if (paths.isEmpty()) emptyList() else checkpoints.buildChangeDetails(projectId, workspace, paths)
        }

    override suspend fun undoFileChange(projectId: String, path: String): Boolean = withContext(Dispatchers.IO) {
        if (checkpoints.isInternalRuntimePath(path) || path !in checkpoints.readChangedPaths(projectId)) return@withContext false
        val workspace = checkpoints.ensureWorkspace(projectId)
        val backup = File(checkpoints.checkpointDir(projectId), "project")
        val target = checkpoints.safeWorkspaceFile(workspace, path)
        val original = checkpoints.safeWorkspaceFile(backup, path)
        if (original.isFile) {
            target.parentFile?.mkdirs()
            original.copyTo(target, overwrite = true)
        } else if (target.isFile) {
            target.delete()
        }
        checkpoints.removeChangedPath(projectId, path)
        true
    }

    override suspend fun acceptFileChange(projectId: String, path: String): Boolean = withContext(Dispatchers.IO) {
        if (checkpoints.isInternalRuntimePath(path) || path !in checkpoints.readChangedPaths(projectId)) return@withContext false
        val workspace = checkpoints.ensureWorkspace(projectId)
        val backup = File(checkpoints.checkpointDir(projectId), "project")
        val current = checkpoints.safeWorkspaceFile(workspace, path)
        val baseline = checkpoints.safeWorkspaceFile(backup, path)
        if (current.isFile) {
            baseline.parentFile?.mkdirs()
            current.copyTo(baseline, overwrite = true)
        } else if (baseline.isFile) {
            baseline.delete()
        }
        checkpoints.removeChangedPath(projectId, path)
        true
    }
}

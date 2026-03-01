package io.t6x.dust.embeddings

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.core.EmbeddingService
import io.t6x.dust.core.EmbeddingStatus
import io.t6x.dust.core.SessionPriority
import io.t6x.dust.onnx.MemoryPressureLevel
import io.t6x.dust.onnx.ONNXConfig
import io.t6x.dust.onnx.ONNXSession
import io.t6x.dust.onnx.ONNXSessionManager
import java.util.concurrent.locks.ReentrantLock
import kotlin.concurrent.withLock

class EmbeddingSessionManager(
    private val onnxSessionManager: ONNXSessionManager,
    private val sessionFactory: (
        onnxSession: ONNXSession,
        modelId: String,
        config: EmbeddingSessionConfig,
        vocabPath: String,
        mergesPath: String?,
        priority: SessionPriority,
    ) -> EmbeddingSession = { onnxSession, modelId, config, vocabPath, mergesPath, _ ->
        val tokenizer = TokenizerFactory.create(
            type = config.tokenizerType,
            vocabPath = vocabPath,
            mergesPath = mergesPath,
        )
        EmbeddingSession(
            sessionId = modelId,
            tokenizer = tokenizer,
            onnxSession = onnxSession,
            config = config,
        )
    },
) : EmbeddingService {
    private val lock = ReentrantLock()
    private val cachedSessions = mutableMapOf<String, CachedSession>()
    private val configs = mutableMapOf<String, EmbeddingSessionConfig>()

    fun loadModel(
        modelPath: String,
        modelId: String,
        vocabPath: String,
        mergesPath: String?,
        config: EmbeddingSessionConfig,
        onnxConfig: ONNXConfig,
        priority: SessionPriority,
    ): EmbeddingSession = lock.withLock {
        val cached = cachedSessions[modelId]
        if (cached != null) {
            cached.refCount += 1
            cached.lastAccessTime = System.nanoTime()
            return cached.session
        }

        val onnxSession = onnxSessionManager.loadModel(modelPath, modelId, onnxConfig, priority)
        val session = sessionFactory(
            onnxSession,
            modelId,
            config,
            vocabPath,
            mergesPath,
            priority,
        )

        cachedSessions[modelId] = CachedSession(
            session = session,
            priority = priority,
            refCount = 1,
            lastAccessTime = System.nanoTime(),
        )
        configs[modelId] = config
        session
    }

    fun loadGGUFModel(
        modelId: String,
        engine: GGUFEmbeddingEngine,
        config: EmbeddingSessionConfig,
    ): EmbeddingSession = lock.withLock {
        val cached = cachedSessions[modelId]
        if (cached != null) {
            cached.refCount += 1
            cached.lastAccessTime = System.nanoTime()
            engine.close()
            return cached.session
        }

        val session = EmbeddingSession(
            sessionId = modelId,
            tokenizer = NoopTokenizer(),
            onnxSession = null,
            config = config,
            ggufEngine = engine,
        )

        cachedSessions[modelId] = CachedSession(
            session = session,
            priority = SessionPriority.INTERACTIVE,
            refCount = 1,
            lastAccessTime = System.nanoTime(),
        )
        configs[modelId] = config
        session
    }

    fun unloadModel(id: String) = lock.withLock {
        val cached = cachedSessions[id]
            ?: throw DustCoreError.ModelNotFound
        if (cached.refCount <= 0) {
            throw DustCoreError.ModelNotFound
        }

        cached.refCount -= 1
        cached.lastAccessTime = System.nanoTime()
    }

    suspend fun forceUnloadModel(id: String) {
        val session = lock.withLock {
            configs.remove(id)
            cachedSessions.remove(id)?.session
        } ?: throw DustCoreError.ModelNotFound

        if (session.usesGGUFBackend) {
            session.close()
            return
        }

        try {
            onnxSessionManager.forceUnloadModel(id)
        } catch (error: DustCoreError) {
            if (error !is DustCoreError.ModelNotFound) {
                throw error
            }
        } finally {
            session.detach(evicted = false)
        }
    }

    fun session(id: String): EmbeddingSession? = lock.withLock {
        val cached = cachedSessions[id] ?: return null
        cached.lastAccessTime = System.nanoTime()
        cached.session
    }

    fun allModelIds(): List<String> = lock.withLock { cachedSessions.keys.sorted() }

    suspend fun evictUnderPressure(level: MemoryPressureLevel) {
        val evicted = lock.withLock {
            val eligible = cachedSessions.filter { (_, cached) ->
                cached.refCount == 0 && when (level) {
                    MemoryPressureLevel.STANDARD -> cached.priority == SessionPriority.BACKGROUND
                    MemoryPressureLevel.CRITICAL -> true
                }
            }
            val sorted = eligible.entries.sortedBy { it.value.lastAccessTime }
            val removed = sorted.map { it.key to it.value.session }
            for ((id, _) in removed) {
                cachedSessions.remove(id)
                configs.remove(id)
            }
            removed
        }

        for ((id, session) in evicted) {
            if (session.usesGGUFBackend) {
                session.evict()
            } else {
                onnxSessionManager.evict(id)
                session.detach(evicted = true)
            }
        }
    }

    override suspend fun embed(texts: List<String>): List<List<Float>> {
        val session = lock.withLock { cachedSessions.values.firstOrNull()?.session }
            ?: throw DustCoreError.ModelNotFound
        return session.embedBatch(texts).map { it.embedding }
    }

    override fun embeddingDimension(): Int = lock.withLock { configs.values.firstOrNull()?.dims ?: 0 }

    override fun status(): EmbeddingStatus = lock.withLock {
        if (cachedSessions.isEmpty()) {
            EmbeddingStatus.IDLE
        } else {
            EmbeddingStatus.READY
        }
    }

    fun refCount(id: String): Int = lock.withLock { cachedSessions[id]?.refCount ?: 0 }

    fun hasCachedSession(id: String): Boolean = lock.withLock { cachedSessions.containsKey(id) }

    val sessionCount: Int
        get() = lock.withLock { cachedSessions.size }
}

private data class CachedSession(
    val session: EmbeddingSession,
    val priority: SessionPriority,
    var refCount: Int,
    var lastAccessTime: Long,
)

private class NoopTokenizer : EmbeddingTokenizer {
    override val vocabSize: Int = 0

    override fun tokenize(text: String, maxLength: Int): TokenizerOutput =
        TokenizerOutput(emptyList(), emptyList(), emptyList())

    override fun countTokens(text: String): Int = 0
}

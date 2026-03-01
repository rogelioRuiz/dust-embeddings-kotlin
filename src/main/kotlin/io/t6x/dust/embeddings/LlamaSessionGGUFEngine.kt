package io.t6x.dust.embeddings

import io.t6x.dust.core.DustCoreError
import io.t6x.dust.llm.LlamaSession

class LlamaSessionGGUFEngine(
    private var session: LlamaSession?,
    private val onRelease: (() -> Unit)? = null,
) : GGUFEmbeddingEngine {
    override fun embed(text: String): FloatArray {
        val current = session ?: throw DustCoreError.SessionClosed
        return current.getEmbedding(text)
    }

    override fun countTokens(text: String): Int {
        val current = session ?: throw DustCoreError.SessionClosed
        return current.countTokens(text)
    }

    override val dims: Int
        get() = session?.embeddingDims ?: 0

    override fun close() {
        release()
    }

    override fun evict() {
        release()
    }

    private fun release() {
        if (session != null) {
            session = null
            onRelease?.invoke()
        }
    }
}

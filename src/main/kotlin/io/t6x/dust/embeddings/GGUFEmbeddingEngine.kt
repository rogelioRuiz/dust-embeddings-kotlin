package io.t6x.dust.embeddings

interface GGUFEmbeddingEngine {
    fun embed(text: String): FloatArray

    fun countTokens(text: String): Int

    val dims: Int

    fun close()

    fun evict()
}

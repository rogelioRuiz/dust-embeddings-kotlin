package io.t6x.dust.embeddings

import kotlin.math.sqrt

object VectorMath {
    fun l2Normalize(vector: MutableList<Float>) {
        val norm = l2Norm(vector)
        if (norm < 1e-12f) {
            return
        }

        for (index in vector.indices) {
            vector[index] /= norm
        }
    }

    fun l2Normalized(vector: List<Float>): List<Float> {
        val normalized = vector.toMutableList()
        l2Normalize(normalized)
        return normalized
    }

    fun cosineSimilarity(a: List<Float>, b: List<Float>): Float {
        if (a.size != b.size || a.isEmpty()) {
            return 0f
        }

        val normA = l2Norm(a)
        val normB = l2Norm(b)
        if (normA < 1e-12f || normB < 1e-12f) {
            return 0f
        }

        var dot = 0f
        for (index in a.indices) {
            dot += a[index] * b[index]
        }
        return dot / (normA * normB)
    }

    private fun l2Norm(vector: List<Float>): Float {
        var sum = 0f
        for (value in vector) {
            sum += value * value
        }
        return sqrt(sum)
    }
}

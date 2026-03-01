package io.t6x.dust.embeddings

import org.junit.Assert.assertEquals
import org.junit.Test

class VectorMathTest {
    @Test
    fun e2T4L2NormalizedReturnsUnitVector() {
        val normalized = VectorMath.l2Normalized(listOf(3f, 4f))

        assertEquals(0.6f, normalized[0], 0.0001f)
        assertEquals(0.8f, normalized[1], 0.0001f)
    }

    @Test
    fun e2T5CosineSimilarityMatchesOrthogonalAndIdenticalVectors() {
        assertEquals(0f, VectorMath.cosineSimilarity(listOf(1f, 0f), listOf(0f, 1f)), 0.0001f)
        assertEquals(1f, VectorMath.cosineSimilarity(listOf(2f, 2f), listOf(2f, 2f)), 0.0001f)
    }
}

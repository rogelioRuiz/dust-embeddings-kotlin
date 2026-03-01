package io.t6x.dust.embeddings

import org.junit.Assert.assertEquals
import org.junit.Test

class PoolingTest {
    @Test
    fun e2T1MeanPoolingUsesAttentionMask() {
        val pooled = Pooling.mean(
            hiddenStates = listOf(
                1f, 0f, 1f,
                9f, 9f, 9f,
                2f, 0f, 2f,
            ),
            attentionMask = listOf(1, 0, 1),
            seqLen = 3,
            hiddenDim = 3,
        )

        assertEquals(1.5f, pooled[0], 0.0001f)
        assertEquals(0f, pooled[1], 0.0001f)
        assertEquals(1.5f, pooled[2], 0.0001f)
    }

    @Test
    fun e2T2ClsPoolingReturnsFirstRow() {
        val pooled = Pooling.cls(listOf(1f, 2f, 3f, 4f, 5f, 6f), hiddenDim = 3)

        assertEquals(listOf(1f, 2f, 3f), pooled)
    }

    @Test
    fun e2T3EosPoolingReturnsLastAttendedToken() {
        val pooled = Pooling.eos(
            hiddenStates = listOf(
                1f, 1f,
                2f, 2f,
                3f, 3f,
                9f, 9f,
                10f, 10f,
            ),
            attentionMask = listOf(1, 1, 1, 0, 0),
            seqLen = 5,
            hiddenDim = 2,
        )

        assertEquals(listOf(3f, 3f), pooled)
    }
}

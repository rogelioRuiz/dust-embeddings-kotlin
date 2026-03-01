package io.t6x.dust.embeddings

object Pooling {
    fun mean(
        hiddenStates: List<Float>,
        attentionMask: List<Int>,
        seqLen: Int,
        hiddenDim: Int,
    ): List<Float> {
        val activeTokenCount = attentionMask.count { it != 0 }
        if (activeTokenCount == 0) {
            return List(hiddenDim) { 0f }
        }

        val result = MutableList(hiddenDim) { 0f }
        for (tokenIndex in 0 until seqLen) {
            if (attentionMask[tokenIndex] == 0) {
                continue
            }
            val base = tokenIndex * hiddenDim
            for (hiddenIndex in 0 until hiddenDim) {
                result[hiddenIndex] += hiddenStates[base + hiddenIndex]
            }
        }

        val divisor = activeTokenCount.toFloat()
        for (hiddenIndex in 0 until hiddenDim) {
            result[hiddenIndex] /= divisor
        }
        return result
    }

    fun cls(hiddenStates: List<Float>, hiddenDim: Int): List<Float> {
        if (hiddenStates.size < hiddenDim) {
            return List(hiddenDim) { 0f }
        }
        return hiddenStates.take(hiddenDim)
    }

    fun eos(
        hiddenStates: List<Float>,
        attentionMask: List<Int>,
        seqLen: Int,
        hiddenDim: Int,
    ): List<Float> {
        val lastIndex = attentionMask.indexOfLast { it != 0 }.takeIf { it >= 0 } ?: 0
        return slice(hiddenStates, lastIndex.coerceIn(0, (seqLen - 1).coerceAtLeast(0)), hiddenDim)
    }

    fun lastToken(
        hiddenStates: List<Float>,
        attentionMask: List<Int>,
        seqLen: Int,
        hiddenDim: Int,
    ): List<Float> = eos(hiddenStates, attentionMask, seqLen, hiddenDim)

    fun apply(
        strategy: String,
        hiddenStates: List<Float>,
        attentionMask: List<Int>,
        seqLen: Int,
        hiddenDim: Int,
    ): List<Float> = when (strategy.lowercase()) {
        "mean" -> mean(hiddenStates, attentionMask, seqLen, hiddenDim)
        "cls" -> cls(hiddenStates, hiddenDim)
        "eos" -> eos(hiddenStates, attentionMask, seqLen, hiddenDim)
        "last_token" -> lastToken(hiddenStates, attentionMask, seqLen, hiddenDim)
        else -> mean(hiddenStates, attentionMask, seqLen, hiddenDim)
    }

    private fun slice(hiddenStates: List<Float>, tokenIndex: Int, hiddenDim: Int): List<Float> {
        if (hiddenDim <= 0) {
            return emptyList()
        }

        val start = tokenIndex * hiddenDim
        val end = (start + hiddenDim).coerceAtMost(hiddenStates.size)
        if (start < 0 || start >= end) {
            return List(hiddenDim) { 0f }
        }

        val values = hiddenStates.subList(start, end)
        return if (values.size == hiddenDim) {
            values.toList()
        } else {
            values + List(hiddenDim - values.size) { 0f }
        }
    }
}

package io.t6x.dust.embeddings

data class EmbeddingSessionConfig(
    val dims: Int,
    val maxSequenceLength: Int,
    val tokenizerType: String,
    val pooling: String,
    val normalize: Boolean,
    val inputNames: InputNames = InputNames(),
    val outputName: String = "last_hidden_state",
) {
    data class InputNames(
        val inputIds: String = "input_ids",
        val attentionMask: String = "attention_mask",
        val tokenTypeIds: String = "token_type_ids",
    )
}

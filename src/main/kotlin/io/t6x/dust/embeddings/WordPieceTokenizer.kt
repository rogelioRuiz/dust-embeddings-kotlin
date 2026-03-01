/*
 * Copyright 2026 T6X
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.t6x.dust.embeddings

import java.io.File
import java.text.Normalizer

class WordPieceTokenizer(vocabPath: String) : EmbeddingTokenizer {
    private val vocab: Map<String, Int>

    override val vocabSize: Int

    init {
        val loadedVocab = linkedMapOf<String, Int>()
        File(vocabPath).useLines { lines ->
            lines.forEachIndexed { index, token ->
                loadedVocab[token] = index
            }
        }
        vocab = loadedVocab
        vocabSize = loadedVocab.size
    }

    override fun tokenize(text: String, maxLength: Int): TokenizerOutput {
        if (maxLength <= 0) {
            return TokenizerOutput(emptyList(), emptyList(), emptyList())
        }

        val contentTokenIds = tokenIds(text)
        val inputIds = mutableListOf(CLS_TOKEN_ID)
        inputIds += contentTokenIds
        inputIds += SEP_TOKEN_ID

        if (inputIds.size > maxLength) {
            inputIds.subList(maxLength, inputIds.size).clear()
            inputIds[maxLength - 1] = SEP_TOKEN_ID
        }

        val realTokenCount = inputIds.size
        while (inputIds.size < maxLength) {
            inputIds += PAD_TOKEN_ID
        }

        val attentionMask = MutableList(realTokenCount) { 1 } + MutableList(maxLength - realTokenCount) { 0 }
        val tokenTypeIds = MutableList(maxLength) { 0 }
        return TokenizerOutput(inputIds, attentionMask, tokenTypeIds)
    }

    override fun countTokens(text: String): Int = tokenIds(text).size

    private fun tokenIds(text: String): List<Int> = normalizedWords(text).flatMap(::wordPieceTokenIds)

    private fun normalizedWords(text: String): List<String> {
        val lowercased = text.lowercase()
        val normalized = Normalizer.normalize(lowercased, Normalizer.Form.NFD)
        val stripped = buildString(normalized.length) {
            normalized.forEach { character ->
                if (Character.getType(character) != Character.NON_SPACING_MARK.toInt()) {
                    append(character)
                }
            }
        }
        val cjkSeparated = buildString(stripped.length * 2) {
            stripped.codePoints().forEach { codePoint ->
                if (isCjk(codePoint)) {
                    append(' ')
                    appendCodePoint(codePoint)
                    append(' ')
                } else {
                    appendCodePoint(codePoint)
                }
            }
        }
        return cjkSeparated.split(WHITESPACE_REGEX).filter { it.isNotEmpty() }
    }

    private fun wordPieceTokenIds(word: String): List<Int> {
        if (word.isEmpty()) {
            return emptyList()
        }

        val characters = word.toCharArray()
        val tokenIds = mutableListOf<Int>()
        var start = 0

        while (start < characters.size) {
            var end = minOf(characters.size, start + MAX_PIECE_LENGTH)
            var matchedTokenId: Int? = null

            while (end > start) {
                var piece = String(characters, start, end - start)
                if (start > 0) {
                    piece = "##$piece"
                }
                val tokenId = vocab[piece]
                if (tokenId != null) {
                    matchedTokenId = tokenId
                    break
                }
                end -= 1
            }

            if (matchedTokenId == null) {
                return listOf(UNK_TOKEN_ID)
            }

            tokenIds += matchedTokenId
            start = end
        }

        return tokenIds
    }

    private fun isCjk(codePoint: Int): Boolean = when (codePoint) {
        in 0x3400..0x4DBF,
        in 0x4E00..0x9FFF,
        in 0xF900..0xFAFF,
        in 0x20000..0x2A6DF,
        in 0x2A700..0x2B73F,
        in 0x2B740..0x2B81F,
        in 0x2B820..0x2CEAF,
        in 0x2F800..0x2FA1F,
        -> true
        else -> false
    }

    private companion object {
        const val PAD_TOKEN_ID = 0
        const val UNK_TOKEN_ID = 100
        const val CLS_TOKEN_ID = 101
        const val SEP_TOKEN_ID = 102
        const val MAX_PIECE_LENGTH = 200
        val WHITESPACE_REGEX = Regex("\\s+")
    }
}

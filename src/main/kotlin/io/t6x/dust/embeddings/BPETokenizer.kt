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

import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.File
import java.nio.charset.StandardCharsets

class BPETokenizer(vocabPath: String, mergesPath: String) : EmbeddingTokenizer {
    private val vocab: Map<String, Int>
    private val merges: Map<String, Int>
    private val byteEncoder: Map<Int, String> = makeByteEncoder()

    override val vocabSize: Int

    init {
        val type = object : TypeToken<Map<String, Int>>() {}.type
        val loadedVocab: Map<String, Int> = Gson().fromJson(File(vocabPath).readText(StandardCharsets.UTF_8), type)
        vocab = loadedVocab
        vocabSize = loadedVocab.size

        val loadedMerges = linkedMapOf<String, Int>()
        var rank = 0
        File(mergesPath).forEachLine(StandardCharsets.UTF_8) { line ->
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                return@forEachLine
            }
            val parts = trimmed.split(' ')
            if (parts.size == 2) {
                loadedMerges["${parts[0]} ${parts[1]}"] = rank
                rank += 1
            }
        }
        merges = loadedMerges
    }

    override fun tokenize(text: String, maxLength: Int): TokenizerOutput {
        if (maxLength <= 0) {
            return TokenizerOutput(emptyList(), emptyList(), emptyList())
        }

        val contentTokenIds = tokenIds(text)
        val inputIds = mutableListOf(START_TOKEN_ID)
        inputIds += contentTokenIds
        inputIds += END_TOKEN_ID

        if (inputIds.size > maxLength) {
            inputIds.subList(maxLength, inputIds.size).clear()
            inputIds[maxLength - 1] = END_TOKEN_ID
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

    private fun tokenIds(text: String): List<Int> =
        text.lowercase()
            .split(WHITESPACE_REGEX)
            .filter { it.isNotEmpty() }
            .flatMap(::bpeTokenStrings)
            .map { vocab[it] ?: PAD_TOKEN_ID }

    private fun bpeTokenStrings(word: String): List<String> {
        val encodedPieces = word.toByteArray(StandardCharsets.UTF_8)
            .map { byte -> byteEncoder[byte.toInt() and 0xFF].orEmpty() }
        if (encodedPieces.isEmpty()) {
            return emptyList()
        }

        val symbols = encodedPieces.dropLast(1).toMutableList()
        symbols += encodedPieces.last() + "</w>"

        while (symbols.size > 1) {
            val bestPair = bestMergePair(symbols) ?: break
            merge(symbols, bestPair)
        }

        return symbols
    }

    private fun bestMergePair(symbols: List<String>): Pair<String, String>? {
        var bestPair: Pair<String, String>? = null
        var bestRank = Int.MAX_VALUE

        for (index in 0 until symbols.lastIndex) {
            val pair = symbols[index] to symbols[index + 1]
            val rank = merges["${pair.first} ${pair.second}"] ?: continue
            if (rank < bestRank) {
                bestRank = rank
                bestPair = pair
            }
        }

        return bestPair
    }

    private fun merge(symbols: MutableList<String>, pair: Pair<String, String>) {
        var index = 0
        while (index < symbols.size - 1) {
            if (symbols[index] == pair.first && symbols[index + 1] == pair.second) {
                symbols[index] = symbols[index] + symbols[index + 1]
                symbols.removeAt(index + 1)
            } else {
                index += 1
            }
        }
    }

    private fun makeByteEncoder(): Map<Int, String> {
        val byteValues = mutableListOf<Int>()
        for (value in 33..126) {
            byteValues += value
        }
        for (value in 161..172) {
            byteValues += value
        }
        for (value in 174..255) {
            byteValues += value
        }

        val codePoints = byteValues.toMutableList()
        var nextCodePoint = 256
        for (value in 0..255) {
            if (value !in byteValues) {
                byteValues += value
                codePoints += nextCodePoint
                nextCodePoint += 1
            }
        }

        return byteValues.indices.associate { index ->
            byteValues[index] to String(Character.toChars(codePoints[index]))
        }
    }

    private companion object {
        const val PAD_TOKEN_ID = 0
        const val START_TOKEN_ID = 49406
        const val END_TOKEN_ID = 49407
        val WHITESPACE_REGEX = Regex("\\s+")
    }
}

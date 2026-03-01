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

import org.junit.Assert.assertEquals
import org.junit.Test
import java.nio.file.Paths

class BPETokenizerTest {
    private fun resourcePath(name: String): String {
        val resource = requireNotNull(javaClass.classLoader.getResource(name)) { "Missing test resource: $name" }
        return Paths.get(resource.toURI()).toString()
    }

    private fun makeTokenizer(): BPETokenizer =
        BPETokenizer(
            vocabPath = resourcePath("clip-vocab-mini.json"),
            mergesPath = resourcePath("clip-merges-mini.txt"),
        )

    private fun makeWordPieceTokenizer(): WordPieceTokenizer = WordPieceTokenizer(resourcePath("bert-vocab-mini.txt"))

    @Test
    fun clipTextUsesExpectedIds() {
        val output = makeTokenizer().tokenize("a photo of a cat", maxLength = 77)

        assertEquals(listOf(49406, 320, 1125, 539, 320, 2368, 49407), output.inputIds.take(7))
    }

    @Test
    fun truncatesToSeventySevenAndEndsWithEndToken() {
        val longText = List(80) { "cat" }.joinToString(" ")
        val output = makeTokenizer().tokenize(longText, maxLength = 77)

        assertEquals(77, output.inputIds.size)
        assertEquals(77, output.attentionMask.size)
        assertEquals(77, output.tokenTypeIds.size)
        assertEquals(49407, output.inputIds.last())
    }

    @Test
    fun countTokensMatchesNonSpecialTokens() {
        val wordPieceTokenizer = makeWordPieceTokenizer()
        val wordPieceText = "the quick brown fox"
        val wordPieceOutput = wordPieceTokenizer.tokenize(wordPieceText, maxLength = 8)
        val wordPieceNonSpecialCount = wordPieceOutput.inputIds.count { it != 0 && it != 101 && it != 102 }

        assertEquals(wordPieceNonSpecialCount, wordPieceTokenizer.countTokens(wordPieceText))

        val bpeTokenizer = makeTokenizer()
        val bpeText = "a photo of a cat"
        val bpeOutput = bpeTokenizer.tokenize(bpeText, maxLength = 16)
        val bpeNonSpecialCount = bpeOutput.inputIds.count { it != 0 && it != 49406 && it != 49407 }

        assertEquals(bpeNonSpecialCount, bpeTokenizer.countTokens(bpeText))
    }
}

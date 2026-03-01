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
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Paths

class WordPieceTokenizerTest {
    private fun resourcePath(name: String): String {
        val resource = requireNotNull(javaClass.classLoader.getResource(name)) { "Missing test resource: $name" }
        return Paths.get(resource.toURI()).toString()
    }

    private fun makeTokenizer(): WordPieceTokenizer = WordPieceTokenizer(resourcePath("bert-vocab-mini.txt"))

    @Test
    fun helloWorldUsesKnownIds() {
        val output = makeTokenizer().tokenize("Hello world", maxLength = 8)

        assertEquals(listOf(101, 7592, 2088, 102), output.inputIds.take(4))
    }

    @Test
    fun unaffableUsesExpectedSubwords() {
        val output = makeTokenizer().tokenize("unaffable", maxLength = 8)

        assertEquals(listOf(101, 4895, 4273, 3085, 102), output.inputIds.take(5))
    }

    @Test
    fun truncatesAndKeepsSepAtEnd() {
        val output = makeTokenizer().tokenize("hello world the quick brown fox hello", maxLength = 8)

        assertEquals(8, output.inputIds.size)
        assertEquals(8, output.attentionMask.size)
        assertEquals(8, output.tokenTypeIds.size)
        assertEquals(102, output.inputIds.last())
    }

    @Test
    fun chineseCharactersAreSeparated() {
        val output = makeTokenizer().tokenize("中文", maxLength = 8)
        val nonPadTokens = output.inputIds.filter { it != 0 }

        assertTrue(nonPadTokens.size >= 4)
        assertEquals(listOf(101, 1746, 1861, 102), nonPadTokens.take(4))
    }

    @Test
    fun unknownTokenFallsBackToUnk() {
        val output = makeTokenizer().tokenize("xyzzyplugh", maxLength = 8)

        assertTrue(output.inputIds.contains(100))
    }
}

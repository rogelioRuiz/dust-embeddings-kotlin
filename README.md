<p align="center">
  <img alt="dust" src="assets/dust_banner.png" width="400">
</p>

<p align="center">
  <strong>Device Unified Serving Toolkit</strong><br>
  <a href="https://github.com/rogelioRuiz/dust">dust ecosystem</a> · v0.1.0 · Apache 2.0
</p>

<p align="center">
  <a href="https://github.com/rogelioRuiz/dust/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
  <img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational">
  <img alt="Maven" src="https://img.shields.io/badge/Maven-io.t6x.dust%3Adust--embeddings-blue">
  <a href="https://developer.android.com/studio/releases/platforms"><img alt="API" src="https://img.shields.io/badge/API-26+-green.svg"></a>
  <a href="https://kotlinlang.org"><img alt="Kotlin" src="https://img.shields.io/badge/Kotlin-2.1-purple.svg"></a>
</p>

---

<p align="center">
<strong>dust ecosystem</strong> —
<a href="../capacitor-core/README.md">capacitor-core</a> ·
<a href="../capacitor-llm/README.md">capacitor-llm</a> ·
<a href="../capacitor-onnx/README.md">capacitor-onnx</a> ·
<a href="../capacitor-serve/README.md">capacitor-serve</a> ·
<a href="../capacitor-embeddings/README.md">capacitor-embeddings</a>
<br>
<a href="../dust-core-kotlin/README.md">dust-core-kotlin</a> ·
<a href="../dust-llm-kotlin/README.md">dust-llm-kotlin</a> ·
<a href="../dust-onnx-kotlin/README.md">dust-onnx-kotlin</a> ·
<strong>dust-embeddings-kotlin</strong> ·
<a href="../dust-serve-kotlin/README.md">dust-serve-kotlin</a>
<br>
<a href="../dust-core-swift/README.md">dust-core-swift</a> ·
<a href="../dust-llm-swift/README.md">dust-llm-swift</a> ·
<a href="../dust-onnx-swift/README.md">dust-onnx-swift</a> ·
<a href="../dust-embeddings-swift/README.md">dust-embeddings-swift</a> ·
<a href="../dust-serve-swift/README.md">dust-serve-swift</a>
</p>

---

# dust-embeddings-kotlin

Standalone tokenizers and embedding runtime primitives for on-device text embeddings.

**Version: 0.1.0**

## Overview

`dust-embeddings-kotlin` provides tokenization and embedding generation for Android. It builds on [dust-core-kotlin](../dust-core-kotlin), [dust-onnx-kotlin](../dust-onnx-kotlin), and [dust-llm-kotlin](../dust-llm-kotlin):

- **BPETokenizer** — byte-pair encoding tokenizer with vocab file loading
- **WordPieceTokenizer** — WordPiece tokenizer for BERT-family models
- **EmbeddingSession** — run an embedding model and extract vector representations
- **EmbeddingSessionManager** — manage multiple embedding sessions with lifecycle control
- **PoolingStrategy** — CLS, mean, and max pooling over token embeddings
- **VectorMath** — cosine similarity, dot product, and L2 normalization utilities

## Architecture

```
src/main/kotlin/io/t6x/dust/embeddings/
├── BPETokenizer.kt
├── WordPieceTokenizer.kt
├── EmbeddingSession.kt
├── EmbeddingSessionManager.kt
├── PoolingStrategy.kt
└── VectorMath.kt
```

## Install

### Gradle — local project dependency

```groovy
// settings.gradle
include ':dust-embeddings-kotlin'
project(':dust-embeddings-kotlin').projectDir = new File('../dust-embeddings-kotlin')

// Also include dependencies
include ':dust-core-kotlin'
project(':dust-core-kotlin').projectDir = new File('../dust-core-kotlin')
include ':dust-onnx-kotlin'
project(':dust-onnx-kotlin').projectDir = new File('../dust-onnx-kotlin')

// build.gradle
dependencies {
    implementation project(':dust-embeddings-kotlin')
}
```

### Gradle — Maven (when published)

```groovy
dependencies {
    implementation 'io.t6x.dust:dust-embeddings:0.1.0'
    // transitive: com.google.code.gson:gson:2.11.0
}
```

## Usage

```kotlin
import io.t6x.dust.embeddings.*

// 1. Tokenize text
val tokenizer = BPETokenizer.fromVocabFile("/data/vocab.json")
val tokens = tokenizer.encode("Hello, world!")

// 2. Generate embeddings
val session = EmbeddingSession(modelPath = "/data/embeddings.onnx")
val vector = session.embed("Hello, world!", pooling = PoolingStrategy.MEAN)

// 3. Compare vectors
val similarity = VectorMath.cosineSimilarity(vectorA, vectorB)

// 4. Clean up
session.close()
```

## Test

```bash
./gradlew test    # 20 JUnit tests (6 suites)
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `BPETokenizerTest` | 3 | Encode/decode, special tokens, merges |
| `EmbeddingSessionManagerTest` | 2 | Lifecycle, concurrent access |
| `EmbeddingSessionTest` | 5 | Embed, batch embed, dimension validation |
| `PoolingStrategyTest` | 3 | CLS, mean, max pooling |
| `VectorMathTest` | 2 | Cosine similarity, L2 normalization |
| `WordPieceTokenizerTest` | 5 | Encode/decode, subword splitting, unknown tokens |

No emulator needed — all tests run on the JVM with mocks.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions, and PR guidelines.

## License

Copyright 2026 Rogelio Ruiz Perez. Licensed under the [Apache License 2.0](LICENSE).

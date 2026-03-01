# Contributing to dust-embeddings-kotlin

Thanks for your interest in contributing! This guide will help you get set up and understand our development workflow.

## Prerequisites

- **JDK 17**
- **Android SDK** (API level set in `build.gradle`)
- **Git**
- **dust-core-kotlin** and **dust-onnx-kotlin** cloned as sibling directories (`../dust-core-kotlin`, `../dust-onnx-kotlin`)

## Getting Started

```bash
# Clone all three repos side-by-side
git clone https://github.com/rogelioRuiz/dust-core-kotlin.git
git clone https://github.com/rogelioRuiz/dust-onnx-kotlin.git
git clone https://github.com/rogelioRuiz/dust-embeddings-kotlin.git

cd dust-embeddings-kotlin

# Run tests
./gradlew test
```

## Project Structure

```
src/main/kotlin/io/t6x/dust/embeddings/
  BPETokenizer.kt            # Byte-pair encoding tokenizer
  EmbeddingSession.kt        # Single embedding session
  EmbeddingSessionConfig.kt  # Session configuration
  EmbeddingSessionManager.kt # Session lifecycle and caching
  EmbeddingTokenizer.kt      # Tokenizer protocol
  GGUFEmbeddingEngine.kt     # GGUF-based embedding engine
  LlamaSessionGGUFEngine.kt  # llama.cpp GGUF engine bridge
  Pooling.kt                 # Mean, CLS, and max pooling strategies
  TokenizerFactory.kt        # Tokenizer instantiation
  TokenizerOutput.kt         # Tokenizer result types
  VectorMath.kt              # Cosine similarity, normalization, dot product
  WordPieceTokenizer.kt      # WordPiece tokenizer

src/test/kotlin/io/t6x/dust/embeddings/
  BPETokenizerTest.kt            # 3 tests
  EmbeddingSessionManagerTest.kt # 2 tests
  EmbeddingSessionTest.kt        # 5 tests
  MockONNXEngine.kt              # Test double for ONNXEngine
  PoolingTest.kt                 # 3 tests
  VectorMathTest.kt              # 2 tests
  WordPieceTokenizerTest.kt      # 5 tests
```

## Making Changes

### 1. Create a branch

```bash
git checkout -b feat/my-feature
```

### 2. Make your changes

- Follow existing Kotlin conventions in the codebase
- Add tests for new functionality

### 3. Add the license header

All `.kt` files must include the Apache 2.0 header:

```kotlin
//
// Copyright 2026 T6X
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
```

### 4. Run checks

```bash
./gradlew test      # All 20 tests must pass
./gradlew build     # Clean build
```

### 5. Commit with a conventional message

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add sentence-level chunking for long inputs
fix: correct L2 normalization for zero vectors
docs: update README usage examples
chore(deps): bump dust-onnx-kotlin to 0.2.0
```

### 6. Open a pull request

Push your branch and open a PR against `main`.

## Reporting Issues

- **Bugs**: Open an issue with steps to reproduce
- **Features**: Open an issue describing the use case and proposed API

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

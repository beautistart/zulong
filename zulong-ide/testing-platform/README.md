# Zulong Testing Platform

A CLI testing framework for the Zulong Core extension, providing gRPC-based integration clients and utilities for automated scenarios.

## Overview

The platform enables end-to-end validation of Zulong's core functionality through:

- **gRPC Adapters** – clients for Zulong's gRPC services  
- **Test Harness** – runner, utilities, and type definitions  
- **Spec Files** – JSON instructions for automated test cases  

## Structure

```
testing-platform/
├── adapters/           # gRPC communication adapters
│   ├── grpcAdapter.ts     # Main gRPC adapter implementation
├── harness/            # Test execution framework
│   ├── runner.ts          # Main test runner
│   ├── types.ts           # Type definitions
│   └── utils.ts           # Utility functions
```

## Prerequisites

- **Node.js** ≥ 18 and **npm** ≥ 8  
- **Protocol Buffers** (used for gRPC)  

Generate proto files in the **root Zulong project**:

```bash
npm run protos
```

## Setup

From the root of the Zulong project:

```bash
npm run install:all
npm run protos
```

Then install and build the testing platform:

```bash
cd testing-platform
npm install
npm run build
```

## Running Spec File Tests

Before running specs, make sure the standalone Zulong Core gRPC server (that runs mocks and host gRPC as well) is running:

```bash
npm run test:sca-server
```

Then finally you can run the cli as:

```bash
npm run start:dev <spec-file-or-folder>
```bash

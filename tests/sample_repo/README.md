# Sample Microservice

This repository serves as a sample codebase for testing Software Repository RAG pipelines.

## Architecture Overview

The system consists of a FastAPI authentication microservice, a SQLite storage backend, and a TypeScript frontend client service.

## Authentication Details

Authentication is implemented via JSON Web Tokens (JWT) signed using HMAC-SHA256. Passwords are hash-verified using SHA256 with custom salting.

### Session Handling

Active user sessions are tracked in the `sessions` table and queried using the `active_users` SQL view.

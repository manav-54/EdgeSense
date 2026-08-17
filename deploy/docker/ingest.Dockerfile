# Ingest service. Multi-stage: the build image has a Go toolchain, the runtime
# image has a static binary and nothing else -- no shell, no package manager.
# This is the only internet-facing listener in the stack, so its attack surface
# is worth minimising rather than inheriting from a base distro.
FROM golang:1.24-bookworm AS build

WORKDIR /src

# Dependencies first, so a code change does not re-download the module graph.
COPY services/ingest/go.mod services/ingest/go.sum ./
RUN go mod download

COPY services/ingest/ ./

RUN go vet ./... \
    && CGO_ENABLED=0 GOOS=linux go build \
        -trimpath \
        -ldflags="-s -w" \
        -o /out/ingest ./cmd/ingest

FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=build /out/ingest /ingest

# Distroless nonroot runs as uid 65532.
USER nonroot:nonroot

EXPOSE 8080 9102
ENTRYPOINT ["/ingest"]

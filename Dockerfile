# syntax=docker/dockerfile:1.7
# Static site: nginx serves the page, the wasm runtime and the model. The heavy assets are not
# in git; the build fetches them (web/fetch-assets.sh) so the image is self contained.
FROM --platform=$BUILDPLATFORM alpine:3.20 AS assets
RUN apk add --no-cache curl bash
WORKDIR /site
COPY web/ ./
RUN bash ./fetch-assets.sh

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=assets /site /usr/share/nginx/html
RUN rm -f /usr/share/nginx/html/fetch-assets.sh /usr/share/nginx/html/README.md
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8080/healthz >/dev/null || exit 1

# Portal: build the React bundle, serve it from nginx, proxy /api to the sink.
FROM node:22-alpine AS build

WORKDIR /app

COPY portal/package.json portal/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY portal/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY deploy/docker/portal-nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

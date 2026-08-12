# Multi-stage: build the static Vite bundle, then serve it with nginx,
# which also proxies API paths to the `api` service on the compose network
# — see nginx.conf. Keeps the frontend calling same-origin relative paths
# (/profile, /compare, /runs/...) in both `npm run dev` (Vite's own proxy,
# see vite.config.ts) and this containerized build, with no separate
# API-base-URL configuration needed either way.
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80

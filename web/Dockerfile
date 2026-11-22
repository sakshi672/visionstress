FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ENV VITE_API_PROXY=http://api:8000
CMD ["npm", "run", "dev"]

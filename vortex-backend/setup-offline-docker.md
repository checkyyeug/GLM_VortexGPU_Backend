# Docker离线解决方案

## 方案1: 使用代理
```cmd
# 设置代理环境变量
set HTTP_PROXY=http://your-proxy:port
set HTTPS_PROXY=http://your-proxy:port
docker-compose up -d
```

## 方案2: 预下载镜像
```cmd
# 手动拉取镜像
docker pull ubuntu:22.04
docker pull nvidia/cuda:12.0-runtime-ubuntu22.04

# 重命名镜像
docker tag ubuntu:22.04 local-ubuntu:22.04
docker tag nvidia/cuda:12.0-runtime-ubuntu22.04 local-cuda:12.0-runtime-ubuntu22.04
```

## 方案3: 修改Dockerfile使用本地镜像
```dockerfile
# 在Dockerfile开头添加
FROM local-cuda:12.0-runtime-ubuntu22.04 AS base
# 或者使用标准Ubuntu
FROM local-ubuntu:22.04 AS base
```

## 方案4: 绕过网络检查
```cmd
# 在docker-compose中添加
environment:
  - DOCKER_TLS_CERTDIR=""
  - DOCKER_TLS_VERIFY=1
```

## 方案5: 使用无基础镜像方案
```dockerfile
FROM scratch
COPY --from=builder /usr/local/bin/vortex-backend /
CMD ["vortex-backend"]
```
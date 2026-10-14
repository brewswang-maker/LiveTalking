#!/usr/bin/env bash
# 生成自签 TLS 证书（含服务器 IP 的 SAN），用于解锁浏览器 getUserMedia（麦克风需安全上下文）。
# 用法： SERVER_IP=62.234.29.115 bash deploy/make_selfsigned_cert.sh
# 产物： /etc/nginx/ssl/huadun.crt 与 huadun.key
set -euo pipefail

SERVER_IP="${SERVER_IP:-62.234.29.115}"
SSL_DIR="${SSL_DIR:-/etc/nginx/ssl}"
DAYS="${DAYS:-825}"

mkdir -p "$SSL_DIR"
cd "$SSL_DIR"

openssl req -x509 -nodes -newkey rsa:2048 -days "$DAYS" \
  -keyout huadun.key -out huadun.crt \
  -subj "/C=CN/ST=Beijing/L=Beijing/O=Huadun/CN=${SERVER_IP}" \
  -addext "subjectAltName=IP:${SERVER_IP},DNS:localhost"

chmod 600 huadun.key
echo "✅ 证书已生成："
echo "   $SSL_DIR/huadun.crt"
echo "   $SSL_DIR/huadun.key"
echo
echo "⚠️ 自签证书浏览器会提示‘不安全’，需在访问页面点‘高级→继续前往’信任一次。"
echo "   正式对外建议用域名 + Let's Encrypt（certbot）替换本自签证书。"

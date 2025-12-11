# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within this project, please follow these steps:

1. **Do NOT open a public issue**
2. **Email the maintainers** with details of the vulnerability
3. **Include**:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Best Practices

When deploying this project:

### Environment Variables

- **Never commit `.env` files** to version control
- Use strong, randomly generated secrets:
  ```bash
  # Generate a secure secret
  openssl rand -hex 32
  ```
- Rotate API keys regularly

### Docker Deployment

- Run containers with minimal privileges
- Use Docker secrets for sensitive data in production
- Keep images updated

### Network Security

- Use HTTPS/TLS in production
- Restrict access to admin ports (Langfuse, MinIO console)
- Use a reverse proxy (nginx, Traefik) for production

### API Keys

- Use read-only API keys where possible
- Implement rate limiting
- Monitor for unusual usage patterns

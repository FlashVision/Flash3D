# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Flash3D, please report it responsibly:

1. **Do NOT** open a public issue
2. Email: gaurav14cs17@gmail.com
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 1 week
- **Fix release**: Within 2 weeks for critical issues

## Scope

This policy covers:
- The Flash3D Python package (`flash3d/`)
- CLI tools
- Docker configurations
- CI/CD pipelines

## Best Practices

When using Flash3D:
- Keep dependencies updated
- Use pinned versions in production
- Validate all file inputs (PLY, OBJ files can be malformed)
- Be cautious with `torch.load()` on untrusted checkpoints (use `weights_only=True` when possible)
- Review Docker configurations before deployment

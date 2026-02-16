# Security Considerations

## API Key Authentication

All API requests require an x-api-key header.
Requests without valid keys are rejected.

## Input Validation

- Audio format restricted to MP3
- Only one audio per request
- Base64 decoding validation
- Language whitelist enforced

## Model Protection

Model file is not publicly exposed.
Inference only via authenticated API.

## Future Security Enhancements

- Rate limiting
- Request logging
- IP throttling
- HTTPS enforcement

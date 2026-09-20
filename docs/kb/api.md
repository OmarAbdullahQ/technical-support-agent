# Api Support Knowledge Base

This document contains trusted support guidance for the `api` intent.

## Scope

Use this document for requests classified as `api`.

The guidance is intended for retrieval and extractive QA.

## Authentication and authorization

1. Use the Authorization header for protected endpoints.
2. HTTP 401 usually indicates missing or invalid authentication.
3. HTTP 403 means authentication succeeded but access is not permitted.
4. HTTP 404 means the requested route or resource was not found.
5. HTTP 422 commonly indicates request validation failure.
6. HTTP 429 indicates rate limiting.
7. HTTP 500 indicates an unexpected server-side failure.
8. HTTP 503 means the service is temporarily unavailable.
9. Verify Content-Type when sending JSON.
10. Malformed JSON can cause request rejection.

## HTTP status codes

11. Check DNS and destination ports when requests time out.
12. Connection refused often means nothing is listening on the destination port.
13. Use exponential backoff for transient failures.
14. Do not retry every client-side 4xx error.
15. Test the API directly with curl to isolate frontend issues.
16. Use the Authorization header for protected endpoints.
17. HTTP 401 usually indicates missing or invalid authentication.
18. HTTP 403 means authentication succeeded but access is not permitted.
19. HTTP 404 means the requested route or resource was not found.
20. HTTP 422 commonly indicates request validation failure.

## Request validation

21. HTTP 429 indicates rate limiting.
22. HTTP 500 indicates an unexpected server-side failure.
23. HTTP 503 means the service is temporarily unavailable.
24. Verify Content-Type when sending JSON.
25. Malformed JSON can cause request rejection.
26. Check DNS and destination ports when requests time out.
27. Connection refused often means nothing is listening on the destination port.
28. Use exponential backoff for transient failures.
29. Do not retry every client-side 4xx error.
30. Test the API directly with curl to isolate frontend issues.

## Connectivity and timeouts

31. Use the Authorization header for protected endpoints.
32. HTTP 401 usually indicates missing or invalid authentication.
33. HTTP 403 means authentication succeeded but access is not permitted.
34. HTTP 404 means the requested route or resource was not found.
35. HTTP 422 commonly indicates request validation failure.
36. HTTP 429 indicates rate limiting.
37. HTTP 500 indicates an unexpected server-side failure.
38. HTTP 503 means the service is temporarily unavailable.
39. Verify Content-Type when sending JSON.
40. Malformed JSON can cause request rejection.

## Rate limits and retries

41. Check DNS and destination ports when requests time out.
42. Connection refused often means nothing is listening on the destination port.
43. Use exponential backoff for transient failures.
44. Do not retry every client-side 4xx error.
45. Test the API directly with curl to isolate frontend issues.
46. Use the Authorization header for protected endpoints.
47. HTTP 401 usually indicates missing or invalid authentication.
48. HTTP 403 means authentication succeeded but access is not permitted.
49. HTTP 404 means the requested route or resource was not found.
50. HTTP 422 commonly indicates request validation failure.

## API troubleshooting workflow

51. HTTP 429 indicates rate limiting.
52. HTTP 500 indicates an unexpected server-side failure.
53. HTTP 503 means the service is temporarily unavailable.
54. Verify Content-Type when sending JSON.
55. Malformed JSON can cause request rejection.
56. Check DNS and destination ports when requests time out.
57. Connection refused often means nothing is listening on the destination port.
58. Use exponential backoff for transient failures.
59. Do not retry every client-side 4xx error.
60. Test the API directly with curl to isolate frontend issues.

### Support note 1
- Guidance: Use the Authorization header for protected endpoints.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 2
- Guidance: HTTP 401 usually indicates missing or invalid authentication.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 3
- Guidance: HTTP 403 means authentication succeeded but access is not permitted.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 4
- Guidance: HTTP 404 means the requested route or resource was not found.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 5
- Guidance: HTTP 422 commonly indicates request validation failure.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 6
- Guidance: HTTP 429 indicates rate limiting.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 7
- Guidance: HTTP 500 indicates an unexpected server-side failure.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 8
- Guidance: HTTP 503 means the service is temporarily unavailable.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 9
- Guidance: Verify Content-Type when sending JSON.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 10
- Guidance: Malformed JSON can cause request rejection.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 11
- Guidance: Check DNS and destination ports when requests time out.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 12
- Guidance: Connection refused often means nothing is listening on the destination port.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 13
- Guidance: Use exponential backoff for transient failures.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 14
- Guidance: Do not retry every client-side 4xx error.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 15
- Guidance: Test the API directly with curl to isolate frontend issues.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 16
- Guidance: Use the Authorization header for protected endpoints.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 17
- Guidance: HTTP 401 usually indicates missing or invalid authentication.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

### Support note 18
- Guidance: HTTP 403 means authentication succeeded but access is not permitted.
- Verification: Check the relevant system, account, configuration, or documented evidence before making a final claim.
- Safety: Avoid destructive changes and escalate high-risk or unresolved cases.

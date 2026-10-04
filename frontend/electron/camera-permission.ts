export class CameraPermissionGate {
  private authorizedUntil = 0;

  authorize(now = Date.now(), validityMs = 5_000): void {
    this.authorizedUntil = now + validityMs;
  }

  consume(permission: string, trustedWindow: boolean, now = Date.now()): boolean {
    const wasAuthorized = this.authorizedUntil > now;
    this.authorizedUntil = 0;
    return permission === 'media' && trustedWindow && wasAuthorized;
  }
}

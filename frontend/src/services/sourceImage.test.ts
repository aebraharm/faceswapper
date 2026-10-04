import { describe, expect, it } from 'vitest';
import { getSourceImageContentType } from './sourceImage';

describe('source image MIME detection', () => {
  it.each([
    ['portrait.jpg', 'image/jpeg', 'image/jpeg'],
    ['portrait.jpeg', '', 'image/jpeg'],
    ['portrait.png', 'application/octet-stream', 'image/png'],
    ['portrait.webp', 'image/webp', 'image/webp'],
    ['portrait.JPG', 'image/pjpeg', 'image/jpeg'],
    ['portrait.png', 'image/jpeg', 'image/jpeg'],
  ])('normalizes %s (%s)', (name, type, expected) => {
    expect(getSourceImageContentType({ name, type })).toBe(expected);
  });

  it('rejects file types with no supported MIME or extension', () => {
    expect(getSourceImageContentType({ name: 'portrait.heic', type: 'image/heic' })).toBeNull();
    expect(getSourceImageContentType({ name: 'portrait', type: '' })).toBeNull();
  });
});

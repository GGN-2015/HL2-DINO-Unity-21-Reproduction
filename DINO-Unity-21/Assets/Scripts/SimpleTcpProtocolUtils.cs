using System;

/// <summary>
/// Utilities for the simple_tcp_server length-prefix protocol.
/// L mode uses one negotiation byte ('L') and then 4-byte big-endian length prefixes.
/// </summary>
public static class SimpleTcpProtocolUtils
{
    public const byte FramingStrategyL = (byte)'L';
    public const int LengthPrefixBytes = 4;
    public const int MaxBuffer = 262144;
    public const int MaxResponseBytes = 1024 * 1024;

    public static void WriteUInt32BigEndian(byte[] target, uint value)
    {
        if (target == null || target.Length < LengthPrefixBytes)
        {
            throw new ArgumentException("Target buffer is too small for a 4-byte length prefix.");
        }

        target[0] = (byte)((value >> 24) & 0xff);
        target[1] = (byte)((value >> 16) & 0xff);
        target[2] = (byte)((value >> 8) & 0xff);
        target[3] = (byte)(value & 0xff);
    }

    public static uint ReadUInt32BigEndian(byte[] source)
    {
        if (source == null || source.Length < LengthPrefixBytes)
        {
            throw new ArgumentException("Source buffer is too small for a 4-byte length prefix.");
        }

        return ((uint)source[0] << 24)
            | ((uint)source[1] << 16)
            | ((uint)source[2] << 8)
            | source[3];
    }
}

using System;
using System.Net.Sockets;

/// <summary>
/// Persistent TCP client for simple_tcp_server L mode:
/// send one 'L' negotiation byte, then exchange 4-byte big-endian length-prefixed messages.
/// </summary>
public class SimpleTcpClient : IDisposable
{
    private readonly Socket socket;
    private readonly byte[] lengthPrefixBuffer;

    public bool IsConnected { get; private set; }
    public string LastError { get; private set; } = string.Empty;

    public SimpleTcpClient(string host, int port)
    {
        lengthPrefixBuffer = new byte[SimpleTcpProtocolUtils.LengthPrefixBytes];
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp)
        {
            NoDelay = true,
            SendTimeout = 5000,
            ReceiveTimeout = 5000
        };

        try
        {
            socket.Connect(host, port);
            SendAll(new[] { SimpleTcpProtocolUtils.FramingStrategyL }, 1);
            IsConnected = true;
            LastError = string.Empty;
        }
        catch (Exception ex)
        {
            IsConnected = false;
            LastError = ex.Message;
            Close();
        }
    }

    public byte[] Request(byte[] msg)
    {
        if (!IsConnected) return null;

        try
        {
            SendLengthPrefixedMessage(msg);
        }
        catch (Exception ex)
        {
            LastError = $"Send failed: {ex.Message}";
            Close();
            return null;
        }

        try
        {
            if (!ReceiveExact(lengthPrefixBuffer, SimpleTcpProtocolUtils.LengthPrefixBytes))
            {
                LastError = "Remote endpoint closed the connection while reading the response length.";
                Close();
                return null;
            }

            uint responseLength = SimpleTcpProtocolUtils.ReadUInt32BigEndian(lengthPrefixBuffer);
            if (responseLength > SimpleTcpProtocolUtils.MaxResponseBytes)
            {
                LastError = $"Response is too large: {responseLength} bytes.";
                Close();
                return null;
            }

            byte[] response = new byte[responseLength];
            if (!ReceiveExact(response, response.Length))
            {
                LastError = "Remote endpoint closed the connection while reading the response body.";
                Close();
                return null;
            }

            return response;
        }
        catch (Exception ex)
        {
            LastError = $"Receive failed: {ex.Message}";
            Close();
            return null;
        }
    }

    public void Close()
    {
        IsConnected = false;
        try
        {
            socket.Shutdown(SocketShutdown.Both);
            socket.Close();
        }
        catch { }
    }

    public void Dispose()
    {
        Close();
    }

    private void SendLengthPrefixedMessage(byte[] data)
    {
        int payloadLength = data == null ? 0 : data.Length;
        SimpleTcpProtocolUtils.WriteUInt32BigEndian(lengthPrefixBuffer, (uint)payloadLength);
        SendAll(lengthPrefixBuffer, SimpleTcpProtocolUtils.LengthPrefixBytes);
        if (payloadLength > 0)
        {
            SendAll(data, payloadLength);
        }
    }

    private void SendAll(byte[] data, int count)
    {
        int sent = 0;
        while (sent < count)
        {
            int sentNow = socket.Send(data, sent, count - sent, SocketFlags.None);
            if (sentNow <= 0) throw new SocketException();
            sent += sentNow;
        }
    }

    private bool ReceiveExact(byte[] target, int count)
    {
        int received = 0;
        while (received < count)
        {
            int receivedNow = socket.Receive(target, received, count - received, SocketFlags.None);
            if (receivedNow <= 0) return false;
            received += receivedNow;
        }

        return true;
    }
}

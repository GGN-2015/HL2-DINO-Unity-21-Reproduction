using System;
using System.Net.Sockets;

/// <summary>
/// Persistent TCP client for simple_tcp_server L mode:
/// send one 'L' negotiation byte, then exchange 4-byte big-endian length-prefixed messages.
/// </summary>
public class SimpleTcpClient : IDisposable
{
    private static readonly byte[] FramingNegotiationBytes = { SimpleTcpProtocolUtils.FramingStrategyL };

    private readonly Socket socket;
    private readonly byte[] lengthPrefixBuffer;
    private readonly object closeLock = new object();
    private bool isConnected = false;
    private bool isClosed = false;

    public bool IsConnected
    {
        get
        {
            if (!isConnected || isClosed) return false;

            if (!IsSocketStillConnected())
            {
                if (string.IsNullOrEmpty(LastError))
                {
                    LastError = "Remote endpoint closed the connection.";
                }

                Close();
                return false;
            }

            return true;
        }
        private set
        {
            isConnected = value;
        }
    }

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
            SendAll(FramingNegotiationBytes, FramingNegotiationBytes.Length);
            IsConnected = true;
            isClosed = false;
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
        byte[] responseBuffer = null;
        int responseLength = Request(msg, ref responseBuffer);
        if (responseLength < 0) return null;

        byte[] response = new byte[responseLength];
        if (responseLength > 0)
        {
            Buffer.BlockCopy(responseBuffer, 0, response, 0, responseLength);
        }

        return response;
    }

    public int Request(byte[] msg, ref byte[] responseBuffer)
    {
        if (!IsConnected) return -1;

        try
        {
            SendLengthPrefixedMessage(msg);
        }
        catch (Exception ex)
        {
            LastError = $"Send failed: {ex.Message}";
            Close();
            return -1;
        }

        try
        {
            if (!ReceiveExact(lengthPrefixBuffer, SimpleTcpProtocolUtils.LengthPrefixBytes))
            {
                LastError = "Remote endpoint closed the connection while reading the response length.";
                Close();
                return -1;
            }

            uint responseLengthUInt = SimpleTcpProtocolUtils.ReadUInt32BigEndian(lengthPrefixBuffer);
            if (responseLengthUInt > SimpleTcpProtocolUtils.MaxResponseBytes)
            {
                LastError = $"Response is too large: {responseLengthUInt} bytes.";
                Close();
                return -1;
            }

            int responseLength = (int)responseLengthUInt;
            if (responseBuffer == null || responseBuffer.Length < responseLength)
            {
                responseBuffer = new byte[responseLength];
            }

            if (!ReceiveExact(responseBuffer, responseLength))
            {
                LastError = "Remote endpoint closed the connection while reading the response body.";
                Close();
                return -1;
            }

            return (int)responseLength;
        }
        catch (Exception ex)
        {
            LastError = $"Receive failed: {ex.Message}";
            Close();
            return -1;
        }
    }

    public void Close()
    {
        lock (closeLock)
        {
            if (isClosed) return;

            IsConnected = false;
            isClosed = true;
            try
            {
                socket.Shutdown(SocketShutdown.Both);
            }
            catch { }

            try
            {
                socket.Close();
            }
            catch { }
        }
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

    private bool IsSocketStillConnected()
    {
        try
        {
            return !(socket.Poll(0, SelectMode.SelectRead) && socket.Available == 0);
        }
        catch (Exception ex)
        {
            LastError = ex.Message;
            return false;
        }
    }
}

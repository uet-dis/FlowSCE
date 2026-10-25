"""Units of the raw features, taken from the extractor documentation of each dataset.

CIC-IDS2018: NetFlow v3 / nProbe field definitions of NF-CICIDS2018-v3. DoHBrw-2020: DoHMeter feature list.
UNSW-NB15: NUSW-NB15_features.csv (Argus / Bro-IDS). A feature whose unit is not stated by the documentation is
recorded as "unknown" with a note; no unit is inferred from the data. One-hot columns are indicators.
"""
NF = "NetFlow v3 / nProbe field definition"
DOH = "DoHMeter feature list (DoHBrw-2020)"
UNSWD = "NUSW-NB15_features.csv"

UNITS = {
    "cic": {
        "L4_SRC_PORT": ("port number", NF), "L4_DST_PORT": ("port number", NF), "PROTOCOL": ("IP protocol number", NF),
        "L7_PROTO": ("nDPI application id", NF), "IN_BYTES": ("bytes", NF), "OUT_BYTES": ("bytes", NF),
        "TCP_FLAGS": ("TCP flag bitmask", NF), "CLIENT_TCP_FLAGS": ("TCP flag bitmask", NF), "SERVER_TCP_FLAGS": ("TCP flag bitmask", NF),
        "MIN_TTL": ("hops (IP TTL)", NF), "MAX_TTL": ("hops (IP TTL)", NF),
        "LONGEST_FLOW_PKT": ("bytes", NF), "SHORTEST_FLOW_PKT": ("bytes", NF), "MIN_IP_PKT_LEN": ("bytes", NF), "MAX_IP_PKT_LEN": ("bytes", NF),
        "SRC_TO_DST_SECOND_BYTES": ("bytes/s", NF), "DST_TO_SRC_SECOND_BYTES": ("bytes/s", NF),
        "RETRANSMITTED_IN_BYTES": ("bytes", NF), "RETRANSMITTED_IN_PKTS": ("packets", NF),
        "RETRANSMITTED_OUT_BYTES": ("bytes", NF), "RETRANSMITTED_OUT_PKTS": ("packets", NF),
        "SRC_TO_DST_AVG_THROUGHPUT": ("bits/s", NF), "DST_TO_SRC_AVG_THROUGHPUT": ("bits/s", NF),
        "NUM_PKTS_UP_TO_128_BYTES": ("packets", NF), "NUM_PKTS_128_TO_256_BYTES": ("packets", NF),
        "NUM_PKTS_256_TO_512_BYTES": ("packets", NF), "NUM_PKTS_512_TO_1024_BYTES": ("packets", NF),
        "NUM_PKTS_1024_TO_1514_BYTES": ("packets", NF),
        "TCP_WIN_MAX_IN": ("bytes (TCP window field, unscaled)", NF), "TCP_WIN_MAX_OUT": ("bytes (TCP window field, unscaled)", NF),
        "ICMP_TYPE": ("ICMP type*256 + code", NF), "ICMP_IPV4_TYPE": ("ICMP type", NF),
        "DNS_QUERY_ID": ("DNS transaction id", NF), "DNS_QUERY_TYPE": ("DNS record type", NF), "DNS_TTL_ANSWER": ("seconds", NF),
        "FTP_COMMAND_RET_CODE": ("FTP reply code", NF),
        "SRC_TO_DST_IAT_MIN": ("milliseconds", NF), "SRC_TO_DST_IAT_MAX": ("milliseconds", NF), "SRC_TO_DST_IAT_AVG": ("milliseconds", NF),
        "SRC_TO_DST_IAT_STDDEV": ("milliseconds", NF), "DST_TO_SRC_IAT_MIN": ("milliseconds", NF), "DST_TO_SRC_IAT_MAX": ("milliseconds", NF),
        "DST_TO_SRC_IAT_AVG": ("milliseconds", NF), "DST_TO_SRC_IAT_STDDEV": ("milliseconds", NF),
    },
    "doh": {
        "SourcePort": ("port number", DOH), "DestinationPort": ("port number", DOH), "Duration": ("seconds", DOH),
        "FlowBytesSent": ("bytes", DOH), "FlowBytesReceived": ("bytes", DOH),
        "FlowSentRate": ("bytes/s", DOH + "; equals FlowBytesSent / Duration on every flow of the subset"),
        "FlowReceivedRate": ("bytes/s", DOH),
        "PacketLengthVariance": ("bytes^2", DOH), "PacketLengthStandardDeviation": ("bytes", DOH), "PacketLengthMean": ("bytes", DOH),
        "PacketLengthMedian": ("bytes", DOH), "PacketLengthMode": ("bytes", DOH),
        "PacketLengthSkewFromMedian": ("dimensionless", DOH), "PacketLengthSkewFromMode": ("dimensionless", DOH),
        "PacketLengthCoefficientofVariation": ("dimensionless", DOH),
        "PacketTimeSkewFromMedian": ("dimensionless", DOH), "PacketTimeSkewFromMode": ("dimensionless", DOH),
        "PacketTimeCoefficientofVariation": ("dimensionless", DOH),
        "ResponseTimeTimeSkewFromMedian": ("dimensionless", DOH), "ResponseTimeTimeSkewFromMode": ("dimensionless", DOH),
        "ResponseTimeTimeCoefficientofVariation": ("dimensionless", DOH),
    },
    "unsw": {
        "sport": ("port number", UNSWD), "dsport": ("port number", UNSWD), "dur": ("seconds", UNSWD + " (record total duration)"),
        "sbytes": ("bytes", UNSWD), "dbytes": ("bytes", UNSWD), "sttl": ("hops (IP TTL)", UNSWD), "dttl": ("hops (IP TTL)", UNSWD),
        "sloss": ("packets", UNSWD), "dloss": ("packets", UNSWD),
        "Sload": ("bits/s", UNSWD), "Dload": ("bits/s", UNSWD), "Spkts": ("packets", UNSWD), "Dpkts": ("packets", UNSWD),
        "swin": ("TCP window advertisement value", UNSWD), "dwin": ("TCP window advertisement value", UNSWD),
        "stcpb": ("TCP base sequence number", UNSWD), "dtcpb": ("TCP base sequence number", UNSWD),
        "smeansz": ("bytes (mean packet size)", UNSWD), "dmeansz": ("bytes (mean packet size)", UNSWD),
        "trans_depth": ("count (pipelined HTTP depth)", UNSWD), "res_bdy_len": ("bytes", UNSWD),
        "Sjit": ("milliseconds", UNSWD), "Djit": ("milliseconds", UNSWD),
        "Sintpkt": ("milliseconds", UNSWD), "Dintpkt": ("milliseconds", UNSWD),
        "is_sm_ips_ports": ("binary flag", UNSWD), "ct_state_ttl": ("count", UNSWD), "ct_flw_http_mthd": ("count", UNSWD),
        "is_ftp_login": ("flag", UNSWD), "ct_ftp_cmd": ("count", UNSWD),
        "ct_srv_src": ("count (last 100 connections)", UNSWD), "ct_srv_dst": ("count (last 100 connections)", UNSWD),
        "ct_dst_ltm": ("count (last 100 connections)", UNSWD), "ct_src_ltm": ("count (last 100 connections)", UNSWD),
        "ct_src_ dltm": ("count (last 100 connections)", UNSWD), "ct_src_dport_ltm": ("count (last 100 connections)", UNSWD),
        "ct_dst_sport_ltm": ("count (last 100 connections)", UNSWD), "ct_dst_src_ltm": ("count (last 100 connections)", UNSWD),
    },
}

# Documented without a unit: recorded as unknown, with the reason.
UNKNOWN = {
    "doh": {n: "DoHMeter lists this time statistic without stating its unit" for n in
            ["PacketTimeVariance", "PacketTimeStandardDeviation", "PacketTimeMean", "PacketTimeMedian", "PacketTimeMode",
             "ResponseTimeTimeVariance", "ResponseTimeTimeStandardDeviation", "ResponseTimeTimeMean", "ResponseTimeTimeMedian",
             "ResponseTimeTimeMode"]},
    "unsw": {"tcprtt": "documented as the sum of synack and ackdat, without a unit",
             "synack": "documented as the SYN to SYN-ACK time, without a unit",
             "ackdat": "documented as the SYN-ACK to ACK time, without a unit"},
}


def unit_of(ds, name):
    """(unit, note) of a raw feature; one-hot columns are indicators."""
    if "=" in name: return "indicator (one-hot)", "one-hot encoding of the categorical field " + name.split("=")[0]
    if name in UNITS.get(ds, {}): return UNITS[ds][name]
    if name in UNKNOWN.get(ds, {}): return "unknown", UNKNOWN[ds][name]
    return "unknown", "not listed with a unit in the extractor documentation"

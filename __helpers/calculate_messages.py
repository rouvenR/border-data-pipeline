#!/usr/bin/env python3

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Calculate message delay and count for a target throughput."
    )
    parser.add_argument(
        "--throughput",
        type=float,
        required=True,
        help="Desired throughput (messages per second)",
    )
    parser.add_argument(
        "--clients",
        type=float,
        required=True,
        help="Desired clients",
    )
    
    args = parser.parse_args()
    
    n_clients = args.clients
    n_subscriber_multiplier = 5
    msg_per_sec = args.throughput
    msg_per_sec_per_client = msg_per_sec / (n_clients * n_subscriber_multiplier)
    delay = int(1000 / msg_per_sec_per_client)
    
    duration_in_sec = 300
    total_messages = int(1000 * duration_in_sec / delay)
    
    throughput_offset = args.throughput - (1000 / delay) * n_clients * n_subscriber_multiplier
    
    print(f"{delay} {total_messages} ; throughput offset: {throughput_offset}")


if __name__ == "__main__":
    main()

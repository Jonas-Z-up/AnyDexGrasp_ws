#!/usr/bin/env python3
"""自研手预塑形测试 — 16 键直达 8 种抓取类型。

  1-8              预塑形 (只做 broadcast, 不闭合)
  Q W E R T Y U I  完整动作 (张开 → 预塑形 → 闭合)
  O                张开手指
  Esc              退出
"""

import os, sys, time

_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _parent)

from hand_bridge import PRESHAPE_LIBRARY, GraspHand

SHOW_KEYS   = {str(n): n for n in range(1, 9)}
EXEC_KEYS   = {"q":1,"w":2,"e":3,"r":4,"t":5,"y":6,"u":7,"i":8}
FINGER_NAMES = {1:"拇指",2:"食指",3:"中指",4:"无名指",5:"小指"}


def print_data(tid):
    p = PRESHAPE_LIBRARY[tid]
    bc = p["broadcast"]
    print(f"\n  Type {tid}: {p['name']}")
    print(f"  预塑形 broadcast:  [{', '.join(str(v) for v in bc)}]")
    print(f"  闭合 close:")
    for m, f, s, pos in p["close"]:
        print(f"    M{m} {FINGER_NAMES.get(m,'?')}:  force={f}g  speed={s}  pos={pos}")


def do_preshape(hand, tid):
    """仅执行预塑形 broadcast, 不闭合, 不预先复位。"""
    p = PRESHAPE_LIBRARY[tid]
    print(f"\n  ▶ Type {tid} {p['name']} — 仅预塑形")
    hand.ctrl.send_broadcast(p["broadcast"])
    time.sleep(p["wait"])
    print(f"  预塑形到位 (未闭合, 按 O 张开)")


def do_grasp(hand, tid):
    """在当前手指状态下执行预塑形 + 闭合, 不预先复位。"""
    p = PRESHAPE_LIBRARY[tid]
    print(f"\n  ▶ Type {tid} {p['name']} — 预塑形 + 闭合")
    hand.ctrl.send_broadcast(p["broadcast"])
    time.sleep(p["wait"])
    for m, f, s, pos in p["close"]:
        hand.ctrl.send_single_motor(m, f, s, pos)
    print(f"  闭合完成 (按 O 张开)")


def print_help():
    print("""
  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
  │  Q  │  W  │  E  │  R  │  T  │  Y  │  U  │  I  │  完整: 张→预塑→闭
  │  1  │  2  │  3  │  4  │  5  │  6  │  7  │  8  │  仅预塑形 broadcast
  └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘

    O = 张开      Esc = 退出
""")


def main():
    print("自研手预塑形测试")
    hand = GraspHand()
    if not hand.connect():
        print("❌ 串口连接失败")
        return 1

    print_help()

    try:
        hand.open()
        time.sleep(1)

        while True:
            try:
                key = input(">>> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if key in SHOW_KEYS:
                tid = SHOW_KEYS[key]
                print_data(tid)
                do_preshape(hand, tid)

            elif key in EXEC_KEYS:
                tid = EXEC_KEYS[key]
                print_data(tid)
                do_grasp(hand, tid)

            elif key == "o":
                hand.open()
                print("  已张开")

            elif key in ("esc", "\x1b"):
                break

            elif key == "?":
                print_help()

            else:
                print(f"  未绑定: {key}  (? 帮助)")

    except KeyboardInterrupt:
        pass
    finally:
        hand.open()
        time.sleep(0.3)
        hand.close_connection()
        print("\n手已复位, 串口关闭")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

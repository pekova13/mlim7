
"""

SOURCE:

    baskets     (68_841_598, [week=90, shopper=100_000, product=300?, price=...])
    coupons     (45_000_000, [week=90, shopper=100_000, product=300?, discount=(10...40)])


MODEL INPUTS as in lecture:
    
    history     (customer, product, week) -> reduced to (c, p, hidden)
    freq        (customer, product, 1)
    coupons     (customer, product, 1)


MODEL as in lecture:

    in1, out1 = encode_decode(history)
    in2, out2 = encode_decode(purchase_frequency)
    in3, out3 = encode_decode(coupons)

    final_in = [in1, out1, in2, out2, in3, out3]

    purchase_prob = softmax(dense(final_in))

"""
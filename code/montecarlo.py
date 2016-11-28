import function
import rand_generator
import random

def pi_estimate(samples, dimensions):

    magnitude = function.Magnitude()
    count = 0
    gen = rand_generator.Generator(magnitude, [dimensions, samples])
    gen.populate_zero_one()
    results = gen.eval()

    for value in results:
        if value < 1:
            count += 1

    if dimensions == 2:
        return count/samples*4
    elif dimensions == 3:
        return count/samples*6
    else:
        print('Hitherto Unconfigure Dimension count')
        return None

def e_estimate(samples):

    total_count = 0

    for i in range(samples):
        cumulative = 0
        count = 0
        while (cumulative < 1):
            cumulative += random.random()
            count += 1
        total_count += count

    return total_count/samples




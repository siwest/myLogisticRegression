import sys
import random
from math import exp, log, sqrt


def read_data(data):
    # Read data file as list-formatted matrix.
    f1 = open(data)
    data = []
    line1 = f1.readline()

    while line1 != '':
        split_line = line1.split()
        temp_line = [1]
        for j in range(0, len(split_line), 1):
            temp_line.append(int(split_line[j]))
        data.append(temp_line)
        line1 = f1.readline()
    f1.close()
    return data


def read_labels(train_labels, length):
    # Read labels file as list-formatted vector.

    def convert_elements_to_int(elements):
        return [int(element) for element in elements]

    classification = [None] * length
    f2 = open(train_labels)
    data_row = f2.readline()
    while data_row != '':
        split_line = convert_elements_to_int(data_row.split())
        classification[split_line[1]] = split_line[0]
        data_row = f2.readline()
    f2.close()
    return classification


def dot_product(vector_1, vector_2):
    # Return sum of element-wise product two vectors.
    # Vectors must have same length, or dot product is undefined
    assert len(vector_1) == len(vector_2)
    result = 0
    for i in range(0, len(vector_1), 1):
        result += vector_1[i] * vector_2[i]
    return result


def initial_theta(length):
    theta = [0] * length
    for j in range(0, length, 1):
        theta[j] = .02 * random.random() - 0.01
    return theta


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def compute_cost(theta, x, y):
    # Return cost for theta given data and labels.
    assert len(x) == len(y)
    m = len(x)
    result = 0

    for i in range(0, m, 1):
        z = dot_product(theta, x[i])
        result += (y[i] * log(sigmoid(z)) + (1 - y[i]) * log(1 - sigmoid(z)))
    result *= -1  # / float(m)

    return result


def gradient_descent(theta, x, y, alpha=0.01, max_iterations=100000, stopping_condition=0.001):
    # Return optimized theta.
    m = len(x)

    result = list(theta)
    for n in range(0, max_iterations, 1):
        result_tmp = list(theta)
        for j in range(0, len(theta), 1):
            delta = 0
            for i in range(0, m, 1):
                if y[i] is not None:
                    z = dot_product(result, x[i])
                    delta += alpha * (y[i] - sigmoid(z)) * x[i][j]
            # print "Delta is ", delta
            result_tmp[j] = result[j] + delta

        # print("Cost of old theta {old_theta} = {old_cost}".format(old_theta=result,
        #                                                           old_cost=compute_cost(result, x, y)))
        #
        # print("Cost of new theta {new_theta} = {new_cost}".format(new_theta=result_tmp,
        #                                                           new_cost=compute_cost(result_tmp, x, y)))

        # if n % (max_iterations // 10) == 0:
        #    print("Cost = {cost}".format(cost=compute_cost(result, x, y)))
        # else:
        #    pass

        if (compute_cost(result, x, y) - compute_cost(result_tmp, x, y)) <= stopping_condition:
            print ("Stopping condition reached after {n}/{max} iterations"
                   .format(n=n, max=max_iterations))
            break
        else:
            pass

        result = result_tmp

    return tuple(result)


def print_normalized_weight(theta):
    print("w = {w}".format(w=theta[1:]))
    normalized_weight = 0

    for element in theta[1:]:
        normalized_weight += element ** 2
        print element

    normalized_weight = sqrt(normalized_weight)
    print("||w|| = {w}".format(w=normalized_weight))

    distance_to_origin = abs(theta[0] / normalized_weight)
    print("distance to origin = {d}".format(d=distance_to_origin))


def predict(classification, theta, data, rows):
    f3 = open('output', 'w')
    for i in range(0, rows, 1):
        if classification[i] is None:
            dp = dot_product(theta, data[i])
            if dp > 0:
                f3.write("1 " + str(i) + "\n")
            else:
                f3.write("0 " + str(i) + "\n")
    print "Prediction complete"
    return 0


def main(path_to_data, path_to_labels):
    # Run logistic regression for specified input files.
    x = read_data(path_to_data)
    y = read_labels(path_to_labels, len(x))
    theta = initial_theta(len(x[0]))
    result = gradient_descent(theta, x, y, alpha=0.01, max_iterations=100000, stopping_condition=0.00000001)
    cost = compute_cost(result, x, y)
    print_normalized_weight(result)
    # print("Theta = {result}".format(result=result))  # These are the weights
    print("Cost = {cost}".format(cost=cost))


if __name__ == '__main__':
    main(path_to_data=sys.argv[1], path_to_labels=sys.argv[2])

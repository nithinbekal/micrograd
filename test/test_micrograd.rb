# frozen_string_literal: true

require "test_helper"

class TestMicrograd < Minitest::Test
  include Micrograd

  def test_simple_arithmetic_with_values
    v1 = Value.new(3.0)
    v2 = Value.new(5.0)

    sum = v1 + v2
    assert_equal 8.0, sum.data

    diff = v1 - v2
    assert_equal -2.0, diff.data

    product = v1 * v2
    assert_equal 15.0, product.data

    quotient = v1 / v2
    assert_in_epsilon 0.6, quotient.data
  end

  def test_arithmetic_with_numeric_types
    v1 = Value.new(3.0)

    sum = v1 + 5.0
    assert_equal 8.0, sum.data

    diff = v1 - 5.0
    assert_equal -2.0, diff.data

    product = v1 * 5.0
    assert_equal 15.0, product.data

    quotient = v1 / 5.0
    assert_in_epsilon 0.6, quotient.data
  end

  def test_coercing_numeric_into_values
    v = Value.new(5.0)

    sum = 1 + v
    assert_equal 6.0, sum.data

    diff = 8 - v
    assert_equal 3.0, diff.data

    product = 3 * v
    assert_equal 15.0, product.data

    quotient = 6 / v
    assert_in_epsilon 1.2, quotient.data
  end

  def test_backward_for_addition
    x = Value.new(3.0)
    y = Value.new(5.0)

    z = x + y
    z.grad = -2.0
    z.backward

    assert_equal -2.0, x.grad
    assert_equal -2.0, y.grad
  end

  def test_backward_for_multiplication
    x = Value.new(3.0)
    y = Value.new(5.0)

    z = x * y
    z.grad = 2.0
    z.backward

    assert_equal 10.0, x.grad
    assert_equal 6.0, y.grad
  end

  def test_backward_for_tanh
    x = Value.new(0.8814)

    o = x.tanh
    o.grad = 1.0
    o.backward

    assert_in_epsilon o.data, 0.7071
    assert_in_epsilon x.grad, 0.5
  end

  def test_complex_expression
    a = Value.new(2.0)
    b = Value.new(-3.0)
    c = Value.new(10.0)
    e = a * b
    d = e + c
    f = Value.new(-2.0)

    l = d * f

    l.grad = 1.0
    l.backward

    assert_in_epsilon f.grad, 4.0
    assert_in_epsilon c.grad, -2.0
    assert_in_epsilon a.grad, 6.0
    assert_in_epsilon b.grad, -4.0
  end

  def test_neuron
    # Inputs
    x1 = Value.new(2.0)
    x2 = Value.new(0.0)

    # Weights
    w1 = Value.new(-3.0)
    w2 = Value.new(1.0)

    bias = Value.new(6.881373)

    x1w1 = x1 * w1
    x2w2 = x2 * w2

    x1w1_x2w2 = x1w1 + x2w2
    n = x1w1_x2w2 + bias

    # tanh is the squishing function which limits output to the range -1..1
    output = n.tanh
    assert_in_epsilon output.data, 0.7071

    output.grad = 1
    output.backward

    assert_in_epsilon n.grad, 0.5

    assert_in_epsilon x1w1_x2w2.grad, 0.5
    assert_in_epsilon bias.grad, 0.5

    assert_in_epsilon x1w1.grad, 0.5
    assert_in_epsilon x2w2.grad, 0.5

    assert_in_epsilon x1.grad, -1.5
    assert_in_epsilon w1.grad, 1

    assert_in_epsilon x2.grad, 0.5
    assert_in_epsilon w2.grad, 0
  end

  def test_value_repeated_in_expression
    a = Value.new(3.0)
    b = a + a
    assert_equal 6.0, b.data

    b.grad = 1.0
    b.backward

    assert_in_epsilon a.grad, 2.0
  end

  def test_more_complex_expression_with_repeated_values
    a = Value.new(-2.0)
    b = Value.new(3.0)

    d = a * b
    e = a + b
    f = d * e

    f.grad = 1.0
    f.backward

    assert_in_epsilon d.grad, 1.0
    assert_in_epsilon e.grad, -6.0

    assert_in_epsilon a.grad, -3.0
    assert_in_epsilon b.grad, -8.0
  end
end

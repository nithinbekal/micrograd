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
    assert_equal 0.6, quotient.data
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
    assert_equal 0.6, quotient.data
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
    assert_equal 1.2, quotient.data
  end
end

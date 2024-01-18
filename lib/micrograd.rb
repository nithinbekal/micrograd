# frozen_string_literal: true

require_relative "micrograd/version"

module Micrograd
  class Error < StandardError; end

  class NullOp
    def backward(_anything) = nil
  end

  class AddOp < Struct.new(:x, :y)
    def backward(value)
      x.grad += value.grad
      y.grad += value.grad
    end
  end

  class MulOp < Struct.new(:x, :y)
    def backward(value)
      x.grad += y.data * value.grad
      y.grad += x.data * value.grad
    end
  end

  class PowOp < Struct.new(:x, :n)
    def backward(value)
      x.grad += n * (x ** (n-1)) * value.grad
    end
  end

  class Value
    def initialize(data, op: NullOp.new)
      @data = data
      @grad = 0.0
      @op = op
    end

    attr_reader :data
    attr_accessor :grad

    def +(other)
      other = wrap(other)
      op = AddOp.new(self, other)

      Value.new(@data + other.data, op: op)
    end

    def -(other) = self + (-wrap(other))

    def *(other)
      other = wrap(other)
      op = MulOp.new(self, other)

      Value.new(@data * other.data, op: op)
    end

    def /(other) = self * (wrap(other) ** -1)

    def **(n)
      op = PowOp.new(self, n)
      Value.new(@data**n, op: op)
    end

    def -@ = Value.new(-self.data)

    def backward
      op.backward(self)
    end

    def coerce(other)
      [Value.new(other), self]
    end

    def inspect
      "Value(#{data}, grad: #{grad})"
    end

    private

    attr_reader :op

    def wrap(v)
      v.is_a?(Value) ? v : Value.new(v)
    end
  end
end

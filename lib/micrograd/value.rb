# frozen_string_literal: true

module Micrograd
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

    def *(other)
      other = wrap(other)
      op = MulOp.new(self, other)

      Value.new(@data * other.data, op: op)
    end

    def tanh
      n = Math.tanh(self.data)
      op = TanhOp.new(self)
      Value.new(n, op: op)
    end

    def **(n)
      op = PowOp.new(self, n)
      Value.new(@data**n, op: op)
    end

    def -@ = Value.new(-self.data)

    def -(other) = self + (-wrap(other))
    def /(other) = self * (wrap(other) ** -1)

    def backward
      op.backward(self)
    end

    def coerce(other)
      [Value.new(other), self]
    end

    def inspect = "Value(#{data}, grad: #{grad})"

    private

    attr_reader :op

    def wrap(v)
      v.is_a?(Value) ? v : Value.new(v)
    end
  end
end
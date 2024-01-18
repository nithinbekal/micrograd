# frozen_string_literal: true

require_relative "micrograd/version"
require_relative "micrograd/value"

module Micrograd
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
end

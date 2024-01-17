# frozen_string_literal: true

require_relative "micrograd/version"

module Micrograd
  class Error < StandardError; end

  class Value
    def initialize(data)
      @data = data
    end

    attr_reader :data

    def +(other)
      Value.new(@data + other.data)
    end

    def -(other)
      Value.new(@data - other.data)
    end

    def *(other)
      Value.new(@data * other.data)
    end

    def *(other)
      Value.new(@data * other.data)
    end

    def /(other)
      Value.new(@data / other.data)
    end

    def coerce(other)
      [Value.new(other), self]
    end
  end
end

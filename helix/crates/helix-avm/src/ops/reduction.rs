//! Reduction operations with error tracking.

use helix_core::types::{BoundedTensor, BoundedValue, ErrorMargin};

/// Sum reduction along all elements.
pub fn sum(input: &BoundedTensor) -> BoundedValue<f64> {
    let mut total = 0.0;
    let mut error = 0.0;

    for v in input.data() {
        total += v.value();
        error += v.absolute_error();
    }

    BoundedValue::new(total, ErrorMargin::absolute(error))
}

/// Mean reduction along all elements.
pub fn mean(input: &BoundedTensor) -> BoundedValue<f64> {
    if input.is_empty() {
        return BoundedValue::exact(0.0);
    }

    let n = input.len() as f64;
    let s = sum(input);

    BoundedValue::new(s.value() / n, ErrorMargin::absolute(s.absolute_error() / n))
}

/// Max reduction along all elements.
pub fn max(input: &BoundedTensor) -> BoundedValue<f64> {
    if input.is_empty() {
        return BoundedValue::exact(f64::NEG_INFINITY);
    }

    let mut max_val = f64::NEG_INFINITY;
    let mut max_error = 0.0;

    for v in input.data() {
        if v.value() > max_val {
            max_val = v.value();
            max_error = v.absolute_error();
        }
    }

    BoundedValue::new(max_val, ErrorMargin::absolute(max_error))
}

/// Min reduction along all elements.
pub fn min(input: &BoundedTensor) -> BoundedValue<f64> {
    if input.is_empty() {
        return BoundedValue::exact(f64::INFINITY);
    }

    let mut min_val = f64::INFINITY;
    let mut min_error = 0.0;

    for v in input.data() {
        if v.value() < min_val {
            min_val = v.value();
            min_error = v.absolute_error();
        }
    }

    BoundedValue::new(min_val, ErrorMargin::absolute(min_error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum() {
        let input = BoundedTensor::from_exact(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let result = sum(&input);
        assert_eq!(result.value(), 10.0);
    }

    #[test]
    fn test_mean() {
        let input = BoundedTensor::from_exact(vec![2.0, 4.0, 6.0, 8.0], vec![4]);
        let result = mean(&input);
        assert_eq!(result.value(), 5.0);
    }

    #[test]
    fn test_max_min() {
        let input = BoundedTensor::from_exact(vec![1.0, 5.0, 3.0, 2.0], vec![4]);
        assert_eq!(max(&input).value(), 5.0);
        assert_eq!(min(&input).value(), 1.0);
    }
}

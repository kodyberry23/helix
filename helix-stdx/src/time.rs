use std::time::{Instant, SystemTime};

use once_cell::sync::Lazy;

const SECOND: i64 = 1;
const MINUTE: i64 = 60 * SECOND;
const HOUR: i64 = 60 * MINUTE;
const DAY: i64 = 24 * HOUR;
const MONTH: i64 = 30 * DAY;
const YEAR: i64 = 365 * DAY;

/// Like `std::time::SystemTime::now()` but does not cause a syscall on every invocation.
///
/// There is just one syscall at the start of the program, subsequent invocations are
/// much cheaper and use the monotonic clock instead of trigerring a syscall.
#[inline]
fn now() -> SystemTime {
    static START_INSTANT: Lazy<Instant> = Lazy::new(Instant::now);
    static START_SYSTEM_TIME: Lazy<SystemTime> = Lazy::new(SystemTime::now);

    *START_SYSTEM_TIME + START_INSTANT.elapsed()
}

/// Formats a timestamp into a human-readable relative time string.
///
/// # Arguments
///
/// * `timestamp` - A point in history. Seconds since UNIX epoch (UTC)
/// * `timezone_offset` - Timezone offset in seconds
///
/// # Returns
///
/// A String representing the relative time (e.g., "4 years ago", "11 months from now")
#[inline]
pub fn format_relative_time(timestamp: i64, timezone_offset: i32) -> String {
    let timestamp = timestamp + timezone_offset as i64;
    let now = now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
        + timezone_offset as i64;

    let time_passed = now - timestamp;

    let time_difference = time_passed.abs();

    let (value, unit) = if time_difference >= YEAR {
        let years = time_difference / YEAR;
        (years, if years == 1 { "year" } else { "years" })
    } else if time_difference >= MONTH {
        let months = time_difference / MONTH;
        (months, if months == 1 { "month" } else { "months" })
    } else if time_difference >= DAY {
        let days = time_difference / DAY;
        (days, if days == 1 { "day" } else { "days" })
    } else if time_difference >= HOUR {
        let hours = time_difference / HOUR;
        (hours, if hours == 1 { "hour" } else { "hours" })
    } else if time_difference >= MINUTE {
        let minutes = time_difference / MINUTE;
        (minutes, if minutes == 1 { "minute" } else { "minutes" })
    } else {
        let seconds = time_difference / SECOND;
        (seconds, if seconds == 1 { "second" } else { "seconds" })
    };
    let value = value.to_string();

    // A zero difference is "now", not the future: commit times are whole
    // seconds, so a commit made in the current second lands here.
    let label = if time_passed.is_negative() {
        "from now"
    } else {
        "ago"
    };

    format!("{value} {unit} {label}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unix_now() -> i64 {
        SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
    }

    #[test]
    fn same_second_is_ago_not_from_now() {
        // Whole-second timestamps mean "just committed" is a zero difference.
        assert_eq!(format_relative_time(unix_now(), 0), "0 seconds ago");
    }

    #[test]
    fn past_and_future() {
        let now = unix_now();
        assert_eq!(format_relative_time(now - 3 * DAY, 0), "3 days ago");
        assert_eq!(format_relative_time(now + 2 * HOUR, 0), "2 hours from now");
        assert_eq!(format_relative_time(now - YEAR, 0), "1 year ago");
    }
}

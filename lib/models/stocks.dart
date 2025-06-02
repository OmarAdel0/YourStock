class Stocks {
  List<Interaday>? timeSeries;

  Stocks({
    this.timeSeries,
  });

  factory Stocks.fromMap(Map<String, dynamic> map,{String? function}) {
    final timeSeries = <Interaday>[];
    final timeSeriesData = map['$function'];
    if (timeSeriesData != null) {
      timeSeriesData.forEach((key, value) {
        timeSeries.add(Interaday.fromMap(value));
      });
    }
    return Stocks(
      timeSeries: timeSeries,
    );
  }
}

class Interaday {
  final String open;
  final String high;
  final String low;
  final String close;
  final String volume;

  const Interaday({
    required this.open,
    required this.high,
    required this.low,
    required this.close,
    required this.volume,
  });

  factory Interaday.fromMap(Map<String, dynamic> map) {
    return Interaday(
      open: map['1. open'] as String,
      high: map['2. high'] as String,
      low: map['3. low'] as String,
      close: map['4. close'] as String,
      volume: map['5. volume'] as String,
    );
  }
}
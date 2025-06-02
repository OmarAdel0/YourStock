class SearchSymbol {
  List<BestMatches>? bestMatches;

  SearchSymbol({
    required this.bestMatches,
  });

  SearchSymbol.fromMap(Map<String, dynamic> map) {
    if (map['bestMatches'] != null) {
      bestMatches = <BestMatches>[];
      map['bestMatches'].forEach((v) {
        bestMatches!.add(BestMatches.fromMap(v));
      });
    }
  }
}

class BestMatches {
  String? symbol;
  String? name;
  String? region;

  BestMatches({
    this.symbol,
    this.name,
    this.region,
  });

  factory BestMatches.fromMap(Map<String, dynamic> map) {
    return BestMatches(
      symbol: map['1. symbol'] as String,
      name: map['2. name'] as String,
      region: map['4. region'] as String,
    );
  }
}

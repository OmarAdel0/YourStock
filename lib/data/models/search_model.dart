class Search {
  String? symbol;
  String? name;
  String? region;

  Search({
    this.symbol,
    this.name,
    this.region,
  });

  factory Search.fromMap(Map<String, dynamic> map) {
    return Search(
      symbol: map['1. symbol'] as String,
      name: map['2. name'] as String,
      region: map['4. region'] as String,
    );
  }
}


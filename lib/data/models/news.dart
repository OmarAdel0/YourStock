
class News {
  String? title;
  String? url;
  String? summary;
  String? banner_image;


    News.fromJson (Map<String, dynamic> json){
    title = json['title'];
    url = json['url'];
    summary = json['summary'];
    banner_image = json['banner_image'];
  }
}
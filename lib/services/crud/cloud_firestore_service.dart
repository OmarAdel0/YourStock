import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:yourstock/services/crud/crud_exception.dart';

class CloudDb {
  String userId = FirebaseAuth.instance.currentUser!.uid;
  final CollectionReference collection =
      FirebaseFirestore.instance.collection("usersWatchlist");

  Future<void> createWatchlist() async {
    if (await isWatchlistExist() == false) {
      const tickerList = <String>[];
      final user = <String, dynamic>{"userId": userId, "ticker": tickerList};
      try {
        await collection.doc(userId).set(user);
      } catch (e) {
        throw CouldNotCreateUserDocument();
      }
    }
  }

  Future<bool> isWatchlistExist() async {
    QuerySnapshot querySnapshot =
        await collection.where(FieldPath.documentId, isEqualTo: userId).get();

    if (querySnapshot.size > 0) return true;

    return false;
  }

  Future<void> deleteWatchlist() async {
    collection.doc(userId).delete();
  }

  Future<Map<String, dynamic>> getDocumentData() async {
    try {
      await createWatchlist();
      final documentReference = collection.doc(userId);
      final DocumentSnapshot doc = await documentReference.get();
      final data = doc.data() as Map<String, dynamic>;
      return data;
    } catch (e) {
      throw CouldNotGetData();
    }
  }

  Future<void> addItemToUserData(String key, dynamic vlaue) async {
    await createWatchlist();
    Map<String, dynamic> data = await getDocumentData();
    List<dynamic> tickerList = data[key];
    if (await isValueExist(key, vlaue) == false) {
      tickerList.add(vlaue);
      final documentReference = collection.doc(userId);
      await documentReference.update({key: tickerList});
    } else {
      throw WatchlistItemAlreadyExist();
    }
  }

  Future<void> removeItemFromUserData(String key, dynamic vlaue) async {
    Map<String, dynamic> data = await getDocumentData();
    List<dynamic> tickerList = data[key];
    if (await isValueExist(key, vlaue)) {
      tickerList.remove(vlaue);
      final documentReference = collection.doc(userId);
      await documentReference.update({key: tickerList});
    } else {
      throw CouldNotFindWatchlistItem();
    }
  }

  Future<bool> isValueExist(String key, dynamic vlaue) async {
    final data = await getDocumentData();
    List<dynamic> tickerList = data[key];
    if (tickerList.contains(vlaue)) {
      return true;
    } else {
      return false;
    }
  }
}

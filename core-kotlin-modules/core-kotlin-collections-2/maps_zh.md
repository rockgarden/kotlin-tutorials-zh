# [在 Kotlin 中使用 Map](https://www.baeldung.com/kotlin/maps)

Kotlin 集合

Map

1. 概述
    在本教程中，我们将了解 Kotlin 中的 `Map` 集合类型。我们首先介绍什么是 map 及其特性。然后我们会学习如何创建 maps。文章的其余部分将讨论常见的操作，例如读取条目、修改条目和数据转换。

2. Kotlin 中的 Map
    Map 是计算机科学中的常见数据结构。在其他编程语言中，它们也被称为字典或关联数组。Map 可以存储一组零个或多个键值对。
    每个键在 map 中是唯一的，并且只能与一个值相关联。但相同的值可以与多个键相关联。
    在 Kotlin 中，[`Map` 接口](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-map/)是主要集合类型之一。我们可以声明键和值为任意类型；没有限制：

    ```kotlin
    interface Map<K, out V>
    ```

    在 Kotlin 中，这些键值对称为条目，由 `Entry` 接口表示：

    ```kotlin
    interface Entry<out K, out V>
    ```

    注意，`Map` 实例是不可变的。我们不能在它被创建后添加、删除或更改条目。如果我们需要可变的 map，Kotlin 提供了 `MutableMap` 类型，允许我们在创建后修改条目：

    ```kotlin
    interface MutableMap<K, V> : Map<K, V>
    ```

    Map 对程序员来说是一个强大的工具，因为它们支持快速的读写访问，即使在处理大量数据时也是如此。这是因为键查找和插入通常是通过哈希实现的，这是一个 O(1) 的操作。让我们看看如何在 Kotlin 中使用 map！

3. 构建 Map
    Kotlin 的标准库包含几种 `Map` 实现。两种主要类型是 `LinkedHashMap` 和 `HashMap`。它们之间的主要区别在于，`LinkedHashMap` 在遍历其条目时会维护插入顺序。像 Kotlin 中的任何类一样，你可以使用默认构造函数实例化它们：

    ```kotlin
    val iceCreamInventory = LinkedHashMap<String, Int>()
    iceCreamInventory["Vanilla"] = 24
    ```

    然而，Kotlin 通过集合 API 提供了更高效的方式来创建 map。接下来我们将探讨这些方法。

    1. 工厂函数
        要在一次操作中声明并填充一个 `Map` 实例，我们使用 `mapOf` 和 `mutableMapOf` 函数。它们接受一个 `Pair` 列表，可以通过可变参数传递。我们还可以使用 `to` 中缀运算符来动态创建这些 `Pair`：

        ```kotlin
        val iceCreamInventory = mapOf("Vanilla" to 24, "Chocolate" to 14, "Rocky Road" to 7)
        ```

        `mapOf` 函数返回一个不可变的 `Map` 类型，而 `mutableMapOf` 返回一个可变的 `MutableMap`。只有在我们需要在创建后修改条目时才应该使用后者。

    2. 条件初始化 Map
        `mapOf` 和 `mutableMapOf` 函数非常适合用于一次性初始化带有条目的 map。但是，有时我们希望创建一个 map 并有条件地将条目放入其中。下面是一个例子：假设我们有四个 `Pair`：

        ```kotlin
        val chocolatePair = "Chocolate" to 3
        val strawberryPair = "Strawberry" to 7
        val vanillaPair = "Vanilla" to 5
        val rockyRoadPair = "Rocky Road" to 10
        ```

        同样，我们想要创建一个 map，并将上述 pairs 放入该 map 中。但是，这次我们有一个额外的要求：

        - “Chocolate” 和 “Strawberry” 条目必须始终放入 map。
        - 对于 “Vanilla” 和 “Rocky Road”，仅当它们的值大于 5 时才将其放入 map。

        根据这个要求，“Vanilla” 条目不应被添加到 map 中，因为它的值是 5。因此，预期的 map 应该如下所示：

        ```kotlin
        val expectedMap = mapOf(chocolatePair, strawberryPair, rockyRoadPair)
        ```

        简单地使用 `mapOf` 函数无法解决这个问题。但我们首先检查候选条目，并构建一个只包含所需条目的 `List<Pair>`。然后我们调用 `toMap` 函数将 pairs 列表转换为 map。接下来，我们看一下它是如何工作的：

        ```kotlin
        val map1 = listOfNotNull(chocolatePair,
        strawberryPair,
        vanillaPair.takeIf { it.second > 5 },
        rockyRoadPair.takeIf { it.second > 5 }).toMap()
        assertEquals(expectedMap, map1)
        ```

        值得一提的是，如果 `takeIf` 函数中的谓词检查返回 false，则表达式（例如 `vanillaPair.takeIf { it.second > 5 }`）返回 null。进一步地，`listOfNotNull` 函数会过滤掉所有 null 值。

        或者，我们可以使用内置的 map 构造器——`buildMap` 函数：

        ```kotlin
        val map2 = buildMap {
        put(chocolatePair.first, chocolatePair.second)
        put(strawberryPair.first, strawberryPair.second)
        if (vanillaPair.second > 5) {
            put(vanillaPair.first, vanillaPair.second)
        }
        if (rockyRoadPair.second > 5) {
            put(rockyRoadPair.first, rockyRoadPair.second)
        }
        }
        assertEquals(expectedMap, map2)
        ```

        `buildMap` 函数允许我们执行一系列构造动作来初始化 map，比如 `put`。因此，我们可以使用 `buildMap` 灵活地初始化一个 map。

    3. 使用 Kotlin 函数式 API
        Kotlin 标准库提供了许多有用的 API，使我们能够以非常简洁的方式生成 Maps。例如，假设我们有一个 `IceCreamShipment` 对象列表。每个 `IceCreamShipment` 有一个 flavor（口味）和 quantity（数量）属性：

        ```kotlin
        val shipments = listOf(
        IceCreamShipment("Chocolate", 3),
        IceCreamShipment("Strawberry", 7),
        IceCreamShipment("Vanilla", 5),
        IceCreamShipment("Chocolate", 5),
        IceCreamShipment("Vanilla", 1),
        IceCreamShipment("Rocky Road", 10),
        )
        ```

        我们想从这个列表生成一个库存 map。一种常见的方法是遍历列表。每个运输批次都会为其 flavor 创建或更新 map 条目：

        ```kotlin
        val iceCreamInventory = mutableMapOf<String, Int>()
        for (shipment in shipments){
            val currentQuantity = iceCreamInventory[shipment.flavor] ?: 0
            iceCreamInventory[shipment.flavor] = currentQuantity + shipment.quantity
        }
        ```

        此实现是可以工作的，但更惯用的方法是使用 Kotlin 的函数式 API：

        ```kotlin
        val iceCreamInventory = shipments
        .groupBy({ it.flavor }, { it.quantity })
        .mapValues { it.value.sum() }
        ```

        我们使用 `groupBy` 将 flavors 与其 quantities 关联起来，然后使用 `mapValues` 函数将数量列表减少为单个总和。如果我们知道我们的列表中每个 key（在这种情况下是冰淇淋口味）都没有多个条目，我们可以改用 `map` 或 [`associateBy`](https://www.baeldung.com/kotlin/list-to-map) 函数。

        如果我们发现我们只是使用 `MutableMap` 来初始填充它，通常有更好的方法使用 Kotlin 的函数式 API 来生成它。总的来说，Kotlin 提供了许多有用的方法来完成诸如将集合转换为 map 这样的常见目标。

4. 访ing Map 条目
    我们使用 `get` 方法从 maps 中检索值。Kotlin 还允许使用方括号表示法作为 `get` 方法的简写：

    ```kotlin
    val map = mapOf("Vanilla" to 24)
    assertEquals(24, map.get("Vanilla"))
    assertEquals(24, map["Vanilla"])
    ```

    有一些 getter 方法定义了在给定 key 不存在时的默认行为。`getValue` 方法将在找不到给定 key 时抛出异常：

    ```kotlin
    assertThrows(NoSuchElementException::class.java) { map.getValue("Banana") }
    ```

    `getOrElse` 方法接受一个 lambda 函数，当 key 不在 map 中时，该函数会被执行。lambda 中的最后一个语句用作返回值：

    ```kotlin
    assertEquals(0, map.getOrElse("Banana", { print("Warning: Flavor not found in map"); 0 }))
    ```

    最后，`getOrDefault` 方法在 key 不存在时返回提供的默认值：

    ```kotlin
    assertEquals(0, map.getOrDefault("Banana", 0))
    ```

5. 添加和更新条目
    如果我们使用 `MutableMap`，那么我们可以使用 `put` 方法添加新条目。我们再次可以使用方括号表示法作为简写：

    ```kotlin
    val iceCreamSales = mutableMapOf<String, Int>()
    iceCreamSales.put("Chocolate", 1)
    iceCreamSales["Vanilla"] = 2
    ```

    也可以使用 `putAll` 方法添加多个条目，它接受要添加到 map 的 Pairs 集合。或者，我们可以使用 `+=` 运算符将一个 map 中的所有条目添加到另一个 map 中：

    ```kotlin
    iceCreamSales.putAll(setOf("Strawberry" to 3, "Rocky Road" to 2))
    iceCreamSales += mapOf("Maple Walnut" to 1, "Mint Chocolate" to 4)
    ```

    请注意，上面提到的所有方法都将在 key 已经存在于 map 中时覆盖当前值。如果我们想要更新条目而不是替换，最好的方法是使用 `merge` 方法。例如：

    ```kotlin
    val iceCreamSales = mutableMapOf("Chocolate" to 2)
    iceCreamSales.merge("Chocolate", 1, Int::plus)
    assertEquals(3, iceCreamSales["Chocolate"])
    ```

    `merge` 方法接受一个 key、一个 value 和一个重新映射函数。重新映射函数定义了如果 key 已存在，我们如何合并旧值和新值。在我们的冰淇淋销售案例中，我们只需要将它们相加。

6. 删除条目
    可变 maps 还提供了删除条目的方法。`remove` 方法接受我们想要从 map 中移除的 key 参数。如果 key 不存在，调用 `remove` 不会抛出异常。我们可以选择性地使用减号赋值 (`-=`) 运算符来执行相同的操作：

    ```kotlin
    val map = mutableMapOf("Chocolate" to 14, "Strawberry" to 9)
    map.remove("Strawberry")
    map -= "Chocolate"
    assertNull(map["Strawberry"])
    assertNull(map["Chocolate"])
    ```

    `MutableMap` 接口还定义了一个 `clear` 方法，可以一次性删除 map 的所有条目。

7. 转换 Maps
    像 Kotlin 中的其他集合类型一样，有许多方法可以按我们的应用程序需求转换 maps。让我们看一下一些有用的操作。对于下面显示的所有示例，这是我们的库存 map 中的初始数据：

    ```kotlin
    val inventory = mutableMapOf(
    "Vanilla" to 24,
    "Chocolate" to 14,
    "Strawberry" to 9,
    )
    ```

    1. 过滤
        Kotlin 为 maps 提供了几种过滤方法。为了按条目 key 或 value 进行过滤，分别有 `filterKeys` 和 `filterValues`。如果我们需要同时按两者进行过滤，可以使用 `filter` 方法。以下是按剩余量过滤冰淇淋库存的一个示例：

        ```kotlin
        val lotsLeft = inventory.filterValues { qty -> qty > 10 }
        assertEquals(setOf("Vanilla", "Chocolate"), lotsLeft.keys)
        ```

        `filterValues` 方法将给定的谓词函数应用于 map 中的每个条目值。过滤后的 map 是满足谓词条件的条目的集合。

        如果我们想要过滤出不匹配此条件的条目，我们可以改用 `filterNot` 方法。

    2. 映射
        `map` 方法接受一个转换函数，将每个条目转换为其他东西。它返回映射值的列表：

        ```kotlin
        val asStrings = inventory.map { (flavor, qty) -> "$qty tubs of $flavor" }
        assertTrue(asStrings.containsAll(setOf("24 tubs of Vanilla", "14 tubs of Chocolate", "9 tubs of Strawberry")))
        assertEquals(3, asStrings.size)
        ```

        在这里，我们使用 `map` 方法生成描述我们当前库存的字符串列表。

    3. 使用 forEach
        最后，作为一个例子，我们将使用我们已经学到的知识，并介绍 `forEach` 方法。`forEach` 方法对给定 map 中的每个条目执行一个操作。在一天接收发货和销售冰淇淋之后，我们需要更新我们店铺的库存 map。我们将从 sales map 中减去所有条目，然后将 shipments map 中的所有条目添加回来，以更新每种 flavor 的数量：

        ```kotlin
        val sales = mapOf("Vanilla" to 7, "Chocolate" to 4, "Strawberry" to 5)
        val shipments = mapOf("Chocolate" to 3, "Strawberry" to 7, "Rocky Road" to 5)
        with(inventory) {
            sales.forEach { merge(it.key, it.value, Int::minus) }
            shipments.forEach { merge(it.key, it.value, Int::plus) }
        }
        assertEquals(17, inventory["Vanilla"]) // 24 - 7 + 0
        assertEquals(13, inventory["Chocolate"]) // 14 - 4 + 3
        assertEquals(11, inventory["Strawberry"]) // 9 - 5 + 7
        assertEquals(5, inventory["Rocky Road"]) // 0 - 0 + 5
        ```

        在这里，我们利用了 `with` 作用域函数来保持代码整洁。这个例子展示了即使复杂的操作也可以通过 Kotlin 强大的 API 轻松完成。

8. 结论

    在本文中，我们介绍了如何在 Kotlin 中使用 maps。maps 是编写高效代码的重要工具。有许多可用的方法；这里涵盖的内容只是其中一部分。我们应该始终[参考文档](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-map/#functions)，确保在代码中使用最合适的方法。

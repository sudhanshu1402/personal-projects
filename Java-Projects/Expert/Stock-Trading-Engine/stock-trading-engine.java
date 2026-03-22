import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class StockTradingEngine {
    static class Stock {
        String symbol;
        double price;
        public Stock(String symbol, double price) { this.symbol = symbol; this.price = price; }
    }

    private Map<String, Stock> market = new HashMap<>();
    private Random random = new Random();

    public StockTradingEngine() {
        market.put("AAPL", new Stock("AAPL", 150.0));
        market.put("GOOG", new Stock("GOOG", 2800.0));
    }

    public void simulateMarket() {
        System.out.println("--- Market Ticker ---");
        for (Stock s : market.values()) {
            double change = (random.nextDouble() - 0.5) * 5;
            s.price += change;
            System.out.printf("%s: $%.2f%n", s.symbol, s.price);
        }
    }

    public static void main(String[] args) throws InterruptedException {
        StockTradingEngine engine = new StockTradingEngine();
        for (int i = 0; i < 5; i++) {
            engine.simulateMarket();
            Thread.sleep(1000);
        }
    }
}
